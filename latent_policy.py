
import torch
from torch import nn
from torch.nn import functional as F

from model_utils import getFeedForward


class LatentPolicy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.skill_embeddings = nn.Embedding(self.config.num_skills, self.config.h_dim)

        self.pi_state_encoder = nn.Linear(self.config.state_size, self.config.h_dim)
        self.monitor_state_encoder = nn.Linear(self.config.state_size, self.config.h_dim)

        self.pi_action_embeddings = nn.Embedding(self.config.action_size*self.config.action_dim, self.config.action_embed_dim)
        self.pi_action_pooler = nn.Linear(self.config.action_embed_dim*self.config.action_dim, self.config.h_dim)

        self.monitor_action_embeddings = nn.Embedding(self.config.action_size*self.config.action_dim, self.config.action_embed_dim)
        self.monitor_action_pooler = nn.Linear(self.config.action_embed_dim*self.config.action_dim, self.config.h_dim)

        self.pi_pos_embeddings = nn.Embedding(self.config.skill_len, self.config.h_dim)
        self.monitor_pos_embeddings = nn.Embedding(self.config.skill_len, self.config.h_dim)

        monitor_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_heads_monitor, dim_feedforward=self.config.dim_feedforward_monitor, batch_first=True)
        self.skill_monitor = nn.TransformerEncoder(monitor_layer, num_layers=self.config.num_layers_monitor)
        self.monitor_head = nn.Linear(self.config.h_dim, self.config.num_skills)

        policy_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_heads, dim_feedforward=self.config.dim_feedforward, batch_first=True)
        self.pi = nn.TransformerDecoder(policy_layer, num_layers=self.config.num_layers)
        self.action_head = nn.Linear(self.config.h_dim, self.config.action_size*self.config.action_dim)

        self.chooser = getFeedForward(self.config.state_size, self.config.h_dim, self.config.num_skills, self.config.num_layers_chooser, self.config.dropout_chooser)

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


    def chooseSkill(self, states, stochastic=True, logits=False):
        assert states.dim() == 2 and states.shape[-1] == self.config.state_size

        out = self.chooser(states)*self.config.temp

        if logits:
            return out
        
        if stochastic:
            dist = torch.distributions.Categorical(probs=torch.softmax(out, dim=-1))
            return dist.sample()
        return torch.argmax(out, dim=-1)
    

    def setChooseSkill(self, states, stochastic=True):
        skill = self.chooseSkill(states, stochastic=stochastic)
        self.setSkill(skill)
        return skill


    def policy(self, states, stochastic=True, action_override=None):
        assert self.curr_skill is not None
        assert states.dim() == 2
        assert states.shape[-1] == self.config.state_size
        assert states.shape[0] == self.n_curr
        assert self.t < self.config.skill_len

        # turn the states into embedded sequences
        h_states = self.pi_state_encoder(states)
        h_states = h_states.unsqueeze(1)

        # add states to history
        if self.history is None:
            self.history = h_states
            self.state_history = states.unsqueeze(1)
        else:
            self.history = torch.cat((self.history, h_states), dim=-2)
            self.state_history = torch.cat((self.state_history, states.unsqueeze(1)), dim=-2)

        T = self.history.shape[1]//2 + 1
        pos_encs = self.pi_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)

        curr_hist = self.history.clone()
        curr_hist[:,::2] += pos_encs
        curr_hist[:, 1::2] += pos_encs[:,:-1]

        seq_len = curr_hist.shape[1]
        temporal_mask = torch.full((seq_len, seq_len), float('-inf'), device=states.device)
        for i in range(seq_len):
            temporal_mask[i, :i+1] = 0

        # get the policy
        logits = self.pi(curr_hist, self.curr_skill, tgt_mask=temporal_mask)[:,-1,:].unsqueeze(-2)
        logits = self.action_head(logits)
        logits = logits.view(logits.shape[0], 1, self.config.action_dim, self.config.action_size)
        logits *= self.config.temp
        if action_override is not None:
            print(logits)

        # get the action
        actions = None
        if stochastic:
            dist = torch.distributions.Categorical(probs=torch.softmax(logits, dim=-1))
            actions = dist.sample()
        else:
            actions = torch.argmax(logits, dim=-1)
        if action_override is not None:
            actions = action_override

        # increment the history
        if self.action_history is None:
            self.action_history = actions
        else:
            self.action_history = torch.cat((self.action_history, actions), dim=1)

        hist_actions = self.getActionEmbeddings(actions, self.pi_action_embeddings, self.pi_action_pooler)
        self.history = torch.cat((self.history, hist_actions), dim=-2)
        self.t += 1

        return actions


    def monitor(self, states, actions, probs=False, pad_mask=None):
        assert states.dim() == 3 and states.shape[-1] == self.config.state_size
        assert actions.dim() == 3 and actions.shape[:-1] == states.shape[:2]
        assert actions.shape[-1] == self.config.action_dim

        T = states.shape[1]

        # turn the states into embedded sequences
        h_states = self.monitor_state_encoder(states)
        h_actions = self.getActionEmbeddings(actions, self.monitor_action_embeddings, self.monitor_action_pooler)

        pos_embs = self.monitor_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)

        # interleave the states and actions into a history
        hist = torch.repeat_interleave(h_states + pos_embs, 2, dim=-2)
        hist[:, 1::2] = h_actions + pos_embs

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
        assert actions.shape[:-1] == states.shape[:2]
        assert actions.shape[-1] == self.config.action_dim
        assert dones.shape == actions.shape[:-1]

        T = states.shape[1]

        # turn the states into embedded sequences
        h_states = self.pi_state_encoder(states)
        h_actions = self.getActionEmbeddings(actions, self.pi_action_embeddings, self.pi_action_pooler)

        pos_embs = self.pi_pos_embeddings(torch.arange(T, device=states.device)).unsqueeze(0)

        # interleave the states and actions into a history
        hist = torch.repeat_interleave(h_states + pos_embs, 2, dim=-2)
        hist[:, 1::2] = h_actions + pos_embs

        temporal_mask = torch.full((T*2-1, T*2-1), True, device=states.device)
        for i in range(T*2-1):
            temporal_mask[i, :i+1] = False 

        # get the encoding of the sequence
        latent_skills = self.skill_embeddings(skills).unsqueeze(-2)

        # get the policy
        logits = self.pi(hist[:,:-1], latent_skills, tgt_mask=temporal_mask)[:, ::2, :]
        logits = self.action_head(logits)
        logits = logits.view(*logits.shape[:2], self.config.action_dim, self.config.action_size)
        logits *= self.config.temp
        assert logits.shape[:-1] == actions.shape

        return logits, self.monitor(states, actions, probs=False, pad_mask=dones), self.chooseSkill(states[:,0], logits=True)


    def getOneHotActions(self, actions):
        assert actions.shape[-1] == self.config.action_dim
        return F.one_hot(actions, self.config.action_size).float()
    
    def flattenActions(self, actions):
        return actions.flatten(start_dim=-2, end_dim=-1)
    
    def stackActions(self, actions):
        s1 = actions.shape[:-1]
        return actions.view(*s1, self.config.action_dim, self.config.action_size)
    
    def getActionEmbeddings(self, actions, emb_mondule, pool_module):
        offset = torch.arange(self.config.action_dim, device=actions.device) * self.config.action_size
        embs = emb_mondule(actions + offset)
        vecs = embs.view(*embs.shape[:-2], -1)
        return pool_module(vecs)