import copy
from functools import partial
import statistics as stats
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.maac import MAACCritic
from utils.rl_utils import build_td_lambda_targets
from utils.rl_utils import disable_gradients, enable_gradients, soft_update
import torch as th
from torch.optim import RMSprop


MSELoss = th.nn.MSELoss()


class MAACLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = MAACCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.policies = partial(self.mac.forward, target=False)
        self.target_policies = partial(self.mac.forward, target=True)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.gamma = args.gamma
        self.tau = args.tau
        self.reward_scale = args.reward_scale
        self.soft = args.soft

        self.agent_optimisers = []
        for i in range(self.n_agents):
            agent_optimiser = RMSprop(params=self.agent_params[i], lr=args.lr, 
                                      alpha=args.optim_alpha, eps=args.optim_eps)
            self.agent_optimisers.append(agent_optimiser)  
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr,
                                        alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        rewards = batch["reward"][:, :-1].unsqueeze(0).expand(
            self.n_agents, -1, -1, -1).reshape(
                self.n_agents, batch.batch_size * (batch.max_seq_length-1), -1)
        terminated = terminated[:, :-1].unsqueeze(0).expand(
            self.n_agents, -1, -1, -1).reshape(
                self.n_agents, batch.batch_size * (batch.max_seq_length-1), -1)
        mask = mask[:, :-1].unsqueeze(0).expand(
            self.n_agents, -1, -1, -1).reshape(
                self.n_agents, batch.batch_size * (batch.max_seq_length-1), -1)

        # avail_actions = batch["avail_actions"][:, :-1]

        critic_train_stats = self._train_critic(batch, rewards, terminated, mask, bs, max_t)

        self.mac.init_hidden(batch.batch_size, target=False)
        samp_acs = [None] * self.n_agents
        all_probs = [None] * self.n_agents
        all_log_pis = [None] * self.n_agents
        all_pol_regs = [None] * self.n_agents
        all_pol_ents = [None] * self.n_agents
        # resample the current state's action use the behavior policy
        # note that we need to remove the resample action which is t = batch.max_seq_length
        for t in range(max_t):
            all_agent_rets = self.policies(batch, t=t, return_extras=True,
                                           return_all_probs=True,
                                           return_log_pi=True,
                                           regularize=True, return_entropy=True)
            for i in range(self.n_agents):
                curr_ac_t, probs_t, log_pi_t, pol_regs_t, ent_t = all_agent_rets[i]
                if t > 0:
                    samp_acs[i] = th.cat((samp_acs[i], curr_ac_t), 0)
                    all_probs[i] = th.cat((all_probs[i], probs_t), 0)
                    all_log_pis[i] = th.cat((all_log_pis[i], log_pi_t), 0)
                    # remember that we need to remove the last timestep
                    if t < max_t - 1:
                        all_pol_regs[i].append(pol_regs_t[0])
                        all_pol_ents[i].append(ent_t)
                else:
                    samp_acs[i] = curr_ac_t
                    all_probs[i] = probs_t
                    all_log_pis[i] = log_pi_t
                    all_pol_regs[i] = pol_regs_t
                    all_pol_ents[i] = [ent_t]
        for i in range(self.n_agents):
            all_pol_regs[i] = [th.stack(all_pol_regs[i]).mean()]
        mean_policy_entropy = stats.mean(
            [th.stack(pol_ents).mean().item() for pol_ents in all_pol_ents])
        # construct resample batch, i.e. replace the actions (and actions_onehot) 
        # with the above resample actions
        resample_batch = copy.deepcopy(batch)
        # reshape the next action to match the shape in next batch
        reshaped_resample_acs = None
        for i in range(self.n_agents):
            _reshaped_resample_ac = samp_acs[i].reshape(bs, max_t, -1).unsqueeze(2)
            if i > 0:
                reshaped_resample_acs = th.cat(
                    (reshaped_resample_acs, _reshaped_resample_ac), 2)
            else:
                reshaped_resample_acs = _reshaped_resample_ac
        # construct the resample action onehot according to resample action
        # the shape of resample action onehot also need to match the shape in resample batch
        reshaped_resample_acs_onehot = resample_batch.data.transition_data[
            'actions_onehot'].clone().fill_(0)
        reshaped_resample_acs_onehot.scatter_(3, reshaped_resample_acs, 1)

        reshaped_dict = {'actions': reshaped_resample_acs,
                         'actions_onehot': reshaped_resample_acs_onehot}
        for key in resample_batch.scheme.keys():
            if key in ('actions', 'actions_onehot'):
                resample_batch.data.transition_data[key] = reshaped_dict[key][:, 1:]
            else:
                resample_batch.data.transition_data[key] = resample_batch[key][:, 1:]
        resample_batch.max_seq_length -= 1

        # construct pre batch all_probs and all_log_pis, i.e. remove the first timestep
        for i in range(self.n_agents):
            all_probs[i] = all_probs[i].reshape(bs, max_t, -1)[:, 1:]
            all_probs[i] = all_probs[i].reshape(bs * (max_t - 1), -1)
            all_log_pis[i] = all_log_pis[i].reshape(bs, max_t, -1)[:, 1:]
            all_log_pis[i] = all_log_pis[i].reshape(bs * (max_t - 1), -1)

        grad_norms = []
        pol_losses = []
        advantages = []
        mask_elems = mask.sum().item()
        critic_rets = self.critic(resample_batch, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(
                range(self.n_agents), all_probs, all_log_pis, all_pol_regs, critic_rets):
            curr_agent = self.mac.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            advantages.append((pol_target * mask).sum().item() / mask_elems)
            if self.soft:
                pol_loss = ((log_pi * (
                    log_pi / self.reward_scale - pol_target).detach()) * mask).sum() / mask.sum()
            else:
                pol_loss = ((log_pi * (-pol_target).detach()) * mask).sum() / mask.sum()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward(retain_graph=True)
            enable_gradients(self.critic)

            pol_losses.append(pol_loss.item())

            grad_norm = th.nn.utils.clip_grad_norm_(
                curr_agent.parameters(), 0.5)
            grad_norms.append(grad_norm)
            self.agent_optimisers[a_i].step()
            self.agent_optimisers[a_i].zero_grad()

        if (self.critic_training_steps - self.last_target_update_step) /\
                self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", \
                "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", stats.mean(advantages), t_env)
            self.logger.log_stat("maac_loss", stats.mean(pol_losses), t_env)
            self.logger.log_stat("agent_grad_norm", stats.mean(grad_norms), t_env)
            # self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("policy_entropy", mean_policy_entropy, t_env)
            self.log_stats_t = t_env


    def _train_critic(self, batch, rewards, terminated, mask, bs, max_t):
        """
        Update central critic for all agents
        """
        self.mac.init_hidden(bs, target=True)
        next_acs = [None] * self.n_agents
        next_log_pis = [None] * self.n_agents
        # get the next state's action use the target policy
        # note that we need to remove the next action which is t = 0
        for t in range(max_t):
            all_agent_rets = self.target_policies(
                batch, t=t, return_extras=True, return_log_pi=True)
            for i in range(self.n_agents):
                curr_next_ac_t, curr_next_log_pi_t = all_agent_rets[i]
                if t > 0:
                    next_acs[i] = th.cat((next_acs[i], curr_next_ac_t), 0)
                    next_log_pis[i] = th.cat((next_log_pis[i], curr_next_log_pi_t), 0)
                else:
                    next_acs[i] = curr_next_ac_t
                    next_log_pis[i] = curr_next_log_pi_t
        # construct next batch, i.e. remove the first timestep in old batch,
        # and replace the actions (and actions_onehot) with the above next actions
        next_batch = copy.deepcopy(batch)
        # reshape the next action to match the shape in next batch
        reshaped_next_acs = None
        for i in range(self.n_agents):
            _reshaped_next_ac = next_acs[i].reshape(bs, max_t, -1).unsqueeze(2)
            if i > 0:
                reshaped_next_acs = th.cat((reshaped_next_acs, _reshaped_next_ac), 2)
            else:
                reshaped_next_acs = _reshaped_next_ac
        # construct the next action onehot according to next action
        # the shape of next action onehot also need to match the shape in next batch
        reshaped_next_acs_onehot = next_batch.data.transition_data[
            'actions_onehot'].clone().fill_(0)
        reshaped_next_acs_onehot.scatter_(3, reshaped_next_acs, 1)

        reshaped_dict = {'actions': reshaped_next_acs,
                         'actions_onehot': reshaped_next_acs_onehot}
        for key in next_batch.scheme.keys():
            if key in ('actions', 'actions_onehot'):
                next_batch.data.transition_data[key] = reshaped_dict[key][:, 1:]
            else:
                next_batch.data.transition_data[key] = next_batch[key][:, 1:]
        next_batch.max_seq_length -= 1

        # construct pre batch, i.e. remove the last timestep in old batch,
        pre_batch = copy.deepcopy(batch)
        for key in pre_batch.scheme.keys():
            pre_batch.data.transition_data[key] = pre_batch[key][:, 1:]
        pre_batch.max_seq_length -= 1

        # calculate next_qs
        next_qs = self.target_critic(next_batch)
        # calculate current_qs
        critic_rets = self.critic(pre_batch, regularize=True)

        # construct next batch next_log_pis, i.e. remove the first timestep
        for i in range(self.n_agents):
            next_log_pis[i] = next_log_pis[i].reshape(bs, max_t, -1)[:, 1:]
            next_log_pis[i] = next_log_pis[i].reshape(bs * (max_t - 1), -1)

        # construct mask, i.e. remove the first timestep

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        q_loss = 0
        td_error = 0
        abs_td_errors = []
        q_takens = []
        targets = []
        mask_elems = mask.sum().item()
        for a_i, nq, log_pi, (pq, regs) in zip(
            range(self.n_agents), next_qs, next_log_pis, critic_rets):
            target_q = (rewards[a_i] + self.gamma * nq * (1 - terminated[a_i]))
            td_error += pq - target_q.detach()
            abs_td_errors.append(((
                pq - target_q.detach()).abs() * mask).sum().item() / mask_elems)
            q_takens.append((pq * mask).sum().item() / mask_elems)
            targets.append((target_q * mask).sum().item() / mask_elems)
            if self.soft:
                target_q -= log_pi / self.reward_scale
            masked_td_error = td_error * mask
            q_loss = (masked_td_error ** 2).sum() / mask.sum()
            for reg in regs:
                q_loss += reg # regularizing attention
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.n_agents)
        self.critic_optimiser.step()
        self.critic_training_steps += (max_t - 1)
        self.critic_optimiser.zero_grad()

        running_log["critic_loss"].append(q_loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        running_log["td_error_abs"].append(stats.mean(abs_td_errors))
        running_log["q_taken_mean"].append(stats.mean(q_takens))
        running_log["target_mean"].append(stats.mean(targets))

        return running_log

    def _update_targets(self):
        soft_update(self.target_critic, self.critic, self.tau)
        for i in range(self.n_agents):
            soft_update(self.mac.target_agents[i], self.mac.agents[i], self.tau)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        for i in range(self.n_agents):
            th.save(self.agent_optimisers[i].state_dict(), "{}/agent_{}_opt.th".format(path, i))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        for i in range(self.n_agents):
            self.agent_optimisers[i].load_state_dict(th.load("{}/agent_{}_opt.th".format(path, i), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
