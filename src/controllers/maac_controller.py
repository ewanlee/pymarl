from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn.functional as F
import copy


# In maac multi-agent controller
# each agents has its own actor parameters,
# and shares parts of parameters in critic
class MAACMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = [None] * self.n_agents
        self.target_hidden_states = [None] * self.n_agents

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, target=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, target=target)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, 
                target=False,
                return_extras=False,
                return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        all_agent_outs = [None] * self.n_agents
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if not target:
            for i in range(self.n_agents):
                all_agent_outs[i], self.hidden_states[i] = self.agents[i](
                    agent_inputs[i], self.hidden_states[i])
        else:
            for i in range(self.n_agents):
                all_agent_outs[i], self.target_hidden_states[i] = self.target_agents[i](
                    agent_inputs[i], self.target_hidden_states[i])

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if not return_extras:
                for i in range(self.n_agents):
                    if getattr(self.args, "mask_before_softmax", True):
                        # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                        reshaped_avail_action = avail_actions.permute(1, 0, 2)[i]
                        all_agent_outs[i][reshaped_avail_action == 0] = -1e10
                    
                    all_agent_outs[i] = th.nn.functional.softmax(all_agent_outs[i], dim=-1)
                    
                    if not test_mode:
                        # Epsilon floor
                        epsilon_action_num = all_agent_outs[i].size(-1)
                        if getattr(self.args, "mask_before_softmax", True):
                            # With probability epsilon, we will pick an available action uniformly
                            epsilon_action_num = reshaped_avail_action.sum(dim=1, keepdim=True).float()

                        all_agent_outs[i] = ((1 - self.action_selector.epsilon) * all_agent_outs[i] + \
                                th.ones_like(all_agent_outs[i]) * \
                                    self.action_selector.epsilon/epsilon_action_num)

                        if getattr(self.args, "mask_before_softmax", True):
                            # Zero out the unavailable actions
                            all_agent_outs[i][reshaped_avail_action == 0] = 0.0
                    all_agent_outs[i] = all_agent_outs[i].unsqueeze(0)

                return th.cat(all_agent_outs, 0).permute(1, 0, 2)

            else:
                all_agent_rets = []
                for i in range(self.n_agents):
                    out = all_agent_outs[i]
                    reshaped_avail_action = avail_actions.permute(1, 0, 2)[i]
                    out[reshaped_avail_action == 0] = -1e10

                    probs = th.nn.functional.softmax(out, dim=-1)
                    
                    int_act = th.multinomial(probs, 1)

                    rets = [int_act]

                    if return_log_pi or return_entropy:
                        log_probs = F.log_softmax(out, dim=1)
                    if return_all_probs:
                        rets.append(probs)
                    if return_log_pi:
                        # return log probability of selected action
                        rets.append(log_probs.gather(1, int_act))
                    if regularize:
                        reg = out ** 2
                        reg[reshaped_avail_action == 0] = 0
                        rets.append([reg.sum() / reshaped_avail_action.sum()])
                    if return_entropy:
                        rets.append(-(log_probs * probs).sum(1).mean())
                    if len(rets) == 1:
                        all_agent_rets.append(rets[0])
                    else:
                        all_agent_rets.append(rets)

                return all_agent_rets

    def init_hidden(self, batch_size, target=False):
        # n_agents x batch_size x hidden_dim
        if not target:
            for i in range(self.n_agents):
                self.hidden_states[i] = self.agents[0].init_hidden().unsqueeze(0).expand(
                    batch_size, 1, -1).permute(1, 0, 2)
        else:
            for i in range(self.n_agents):
                self.target_hidden_states[i] = \
                    self.target_agents[0].init_hidden().unsqueeze(0).expand(
                        batch_size, 1, -1).permute(1, 0, 2)

    def parameters(self):
        params = []
        for i in range(self.n_agents):
            params.append(self.agents[i].parameters())
        return params

    def target_parameters(self):
        target_params = []
        for i in range(self.n_agents):
            target_params.append(self.target_agents[i].parameters())
        return target_params

    def load_state(self, other_mac):
        for i in range(self.n_agents):
            self.agents[i].load_state_dict(other_mac.agents[i].state_dict())
            self.target_agents[i].load_state_dict(other_mac.target_agents[i].state_dict())

    def cuda(self):
        for i in range(self.n_agents):
            self.agents[i].cuda()
            self.target_agents[i].cuda()

    def save_models(self, path):
        for i in range(self.n_agents):
            th.save(self.agents[i].state_dict(), "{}/agent-{}.th".format(path, i))
            th.save(self.target_agents[i].state_dict(), "{}/target-agent-{}.th".format(path, i))

    def load_models(self, path):
        for i in range(self.n_agents):
            self.agents[i].load_state_dict(
                th.load("{}/agent-{}.th".format(path, i),
                map_location=lambda storage, loc: storage))
            self.target_agents[i].load_state_dict(
                th.load("{}/target-agent-{}.th".format(path, i),
                map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agents = []
        self.target_agents = []
        for i in range(self.n_agents):
            self.agents.append(agent_REGISTRY[self.args.agent](input_shape, self.args))
            self.target_agents.append(copy.deepcopy(self.agents[i]))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        inputs = th.cat([x.reshape(bs, self.n_agents, -1).permute(1, 0, 2) for x in inputs], dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
