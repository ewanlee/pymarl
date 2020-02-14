import torch as th
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import numpy as np


class MAACCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, scheme, args):
        super(MAACCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.attend_heads = self.args.attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        state_shape, action_shape = self._get_input_shape(scheme)
        # assume the state_shape and action_shape of all agents are identical
        idim = state_shape + action_shape
        odim = action_shape
        for _ in self.n_agents:
            encoder = nn.Sequential()
            if args.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, args.critic_hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * args.critic_hidden_dim,
                                                      args.critic_hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(args.critic_hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if args.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            state_shape, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(state_shape,
                                                            args.critic_hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = args.critic_hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for _ in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(
                args.critic_hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(
                args.critic_hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(
                nn.Linear(args.critic_hidden_dim, attend_dim),
                nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

        # input_shape = self._get_input_shape(scheme)
        # self.output_type = "q"

        # Set up network layers
        # self.fc1 = nn.Linear(input_shape, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, self.n_actions)

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def forward(self, batch, t=None, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            batch (EpisodeBatch): raw trajectory data
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        inps = self._build_inputs(batch, t=t)
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [th.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = th.matmul(selector.view(selector.shape[0], 1, -1),
                                             th.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (th.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = th.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            # if logger is not None:
            #     logger.add_scalars('agent%i/attention' % a_i,
            #                        dict(('head%i_entropy' % h_i, ent) for h_i, ent
            #                             in enumerate(head_entropies)),
            #                        niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

    # def forward(self, batch, t=None):
    #     inputs = self._build_inputs(batch, t=t)
    #     x = F.relu(self.fc1(inputs))
    #     x = F.relu(self.fc2(x))
    #     # batch_size x n_timesteps x n_agents x act_dim
    #     q = self.fc3(x)
    #     return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # batch_size x n_timesteps x n_agents x obs_dim
        inputs_state = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        inputs_action = batch["actions_onehot"][:, ts]
        # n_agents x batch_size x n_timesteps x obs_dim
        inputs_state = inputs_state.permute(2, 0, 1, 3)
        # n_agents x batch_size x n_timesteps x act_dim
        inputs_action = inputs_action.permute(2, 0, 1, 3)
        inputs = [(s, a) for s, a in zip(inputs_state, inputs_action)]
        return inputs

    def _get_input_shape(self, scheme):
        # state
        state_shape = scheme["state"]["vshape"]
        # observation
        state_shape += scheme["obs"]["vshape"]
        # actions and last actions
        state_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # agent id
        state_shape += self.n_agents
        # actions
        action_shape = scheme["actions_onehot"]["vshape"][0]
        return state_shape, action_shape