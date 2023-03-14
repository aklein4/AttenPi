
import torch
from torch import nn
from torch.nn import functional as F

from configs import DefaultLatentPolicy


class LatentPolicy(nn.Module):

    def __init__(self, config=DefaultLatentPolicy):
        super().__init__()
        self.config = config
        
        self.skill_embeddings = nn.Embedding(self.config.num_skills, self.config.h_dim)

        self.pi_state_encoder = nn.Linear(self.config.state_size, self.config.h_dim)
        self.monitor_state_encoder = nn.Linear(self.config.state_size, self.config.h_dim)

        self.pi_action_embeddings = nn.Embedding(self.config.action_size, self.config.h_dim)
        self.monitor_action_embeddings = nn.Embedding(self.config.action_size, self.config.h_dim)

        self.pi_pos_embeddings = nn.Embedding(self.config.max_seq_len, self.config.h_dim)
        self.monitor_pos_embeddings = nn.Embedding(self.config.max_seq_len, self.config.h_dim)

        monitor_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_heads_monitor, dim_feedforward=self.config.dim_feedforward_monitor, batch_first=True)
        self.skill_monitor = nn.TransformerEncoder(monitor_layer, num_layers=self.config.num_layers_monitor)
        self.monitor_head = nn.Linear(self.config.h_dim, self.config.num_skills)

        policy_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_heads, dim_feedforward=self.config.dim_feedforward, batch_first=True)
        self.pi = nn.TransformerDecoder(policy_layer, num_layers=self.config.num_layers)
        self.action_head = nn.Linear(self.config.h_dim, self.config.action_size)

        self.curr_skill = None
        self.n_curr = None
        self.history = None
        self.t = 0


    def setSkill(self, skills):
        assert skills.dim() == 1

        self.curr_skill = self.skill_embeddings(skills).unsqueeze(-2)
        self.n_curr = skills.numel()

        self.history = None
        self.state_history = None
        self.action_history = None
        self.t = torch.tensor([0], device=skills.device)


    def getSkill(self):
        return self.curr_skill.clone().detach()


    def policy(self, states, stochastic=True):
        assert self.curr_skill is not None
        assert states.dim() == 2
        assert states.shape[-1] == self.config.state_size
        assert states.shape[0] == self.n_curr
        assert self.t < self.config.max_seq_len

        # turn the states into embedded sequences
        h_states = self.pi_state_encoder(states) + self.pi_pos_embeddings(self.t)
        h_states = h_states.unsqueeze(1)

        # add states to history
        if self.history is None:
            self.history = h_states
            self.state_history = states.unsqueeze(1)
        else:
            self.history = torch.cat((self.history, h_states), dim=-2)
            self.state_history = torch.cat((self.state_history, states.unsqueeze(1)), dim=-2)

        # get the policy
        logits = self.pi(self.history, self.curr_skill, tgt_mask=None)[:,-1,:].unsqueeze(-2)
        logits = self.action_head(logits)

        # get the action
        actions = None
        if stochastic:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        else:
            actions = torch.argmax(logits, dim=-1)

        # increment the history
        if self.action_history is None:
            self.action_history = actions
        else:
            self.action_history = torch.cat((self.action_history, actions), dim=-1)

        hist_actions = self.pi_action_embeddings(actions) + self.pi_pos_embeddings(self.t)
        self.history = torch.cat((self.history, hist_actions), dim=-2)
        self.t += 1

        return actions


    def monitor(self, states, actions, probs=False, pad_mask=None):
        assert states.dim() == 3 and states.shape[-1] == self.config.state_size
        assert actions.dim() == 2 and actions.shape == states.shape[:2]

        T = states.shape[1]

        # turn the states into embedded sequences
        states = self.monitor_state_encoder(states) + self.monitor_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)

        # turn the actions into embedded sequences
        actions = self.monitor_action_embeddings(actions) + self.monitor_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)

        # interleave the states and actions into a history
        hist = torch.cat((states, actions), dim=-2).clone()
        hist[:, 0::2, :] = states
        hist[:, 1::2, :] = actions

        if pad_mask is not None:
            pad_mask = torch.repeat_interleave(pad_mask, 2, dim=-1)

        # get the encoding of the sequence
        latent_skills = self.skill_monitor(hist, src_key_padding_mask=pad_mask)
        latent_skills = torch.mean(latent_skills, dim=-2)

        # get the predictions
        logits = self.monitor_head(latent_skills)

        if probs:
            return torch.softmax(logits, dim=-1)
        return logits


    def monitorHistory(self, probs=False):
        assert self.state_history is not None
        assert self.action_history is not None

        return self.monitor(self.state_history, self.action_history, probs=probs)


    def forward(self, x):
        states, actions, skills, dones = tuple(x)

        assert skills.dim() == 1
        assert states.dim() == 3 and states.shape[-1] == self.config.state_size
        assert actions.shape == states.shape[:2]
        assert dones.shape == actions.shape

        T = states.shape[1]

        # turn the states into embedded sequences
        h_states = self.pi_state_encoder(states) + self.pi_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)
        h_actions = self.pi_action_embeddings(actions) + self.pi_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)

        # interleave the states and actions into a history
        hist = torch.cat((h_states, h_actions), dim=-2).clone()
        hist[:, 0::2, :] = h_states
        hist[:, 1::2, :] = h_actions

        temporal_mask = torch.full((T*2, T*2), float('-inf'), device=states.device)
        for i in range(T*2):
            temporal_mask[i, :i+1] = 0

        # get the encoding of the sequence
        latent_skills = self.skill_embeddings(skills).unsqueeze(-2)

        # get the policy
        logits = self.pi(hist, latent_skills, tgt_mask=temporal_mask)[:, ::2, :]
        logits = self.action_head(logits)
        assert logits.shape[:2] == actions.shape

        return logits, self.monitor(states, actions, probs=False, pad_mask=dones)
