
import torch
from torch import nn
from torch.nn import functional as F

from model_utils import SkipNet


class LatentPolicy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.skill_embeddings = nn.Embedding(self.config.num_skills, self.config.skill_embed_dim)

        self.monitor_state_encoder = nn.Linear(self.config.state_size, self.config.h_dim)

        self.monitor_action_embeddings = nn.Embedding(self.config.action_size*self.config.action_dim, self.config.action_embed_dim)
        self.monitor_action_pooler = nn.Linear(self.config.action_embed_dim*self.config.action_dim, self.config.h_dim)
        
        self.monitor_pos_embeddings = nn.Embedding(self.config.skill_len, self.config.h_dim)

        monitor_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_heads_monitor, dim_feedforward=self.config.dim_feedforward_monitor, batch_first=True)
        self.skill_monitor = nn.TransformerEncoder(monitor_layer, num_layers=self.config.num_layers_monitor)
        self.monitor_head = nn.Linear(self.config.h_dim, self.config.num_skills)

        self.pi = SkipNet(self.config.state_size+self.config.skill_embed_dim, self.config.h_dim, self.config.action_dim*self.config.action_size, self.config.num_layers, self.config.dropout)

        self.chooser = SkipNet(self.config.state_size, self.config.h_dim, self.config.num_skills, self.config.num_layers, self.config.dropout)

        self.opter = SkipNet(self.config.state_size, self.config.h_dim, self.config.action_dim*self.config.action_size, self.config.num_layers, self.config.dropout)

        self.curr_skill = None
        self.n_curr = None
        self.history = None
        self.t = 0


    def setSkill(self, skills):
        assert skills.dim() == 1

        self.curr_skill = self.skill_embeddings(skills)
        self.n_curr = skills.numel()

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

        if self.state_history is None:
            self.state_history = states.unsqueeze(1)
        else:
            self.state_history = torch.cat((self.state_history, states.unsqueeze(1)), dim=1)

        # get the policy
        logits = self.pi(torch.cat([states, self.curr_skill], dim=-1)) * self.config.temp
        logits = (1-self.config.dagger_beta)*logits + self.config.dagger_beta*self.opter(states)
        logits = self.stackActions(logits)
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
            self.action_history = actions.unsqueeze(1)
        else:
            self.action_history = torch.cat((self.action_history, actions.unsqueeze(1)), dim=1)

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

        # get the encoding of the sequence
        latent_skills = torch.stack([self.skill_embeddings(skills)]*T, dim=1)
        assert latent_skills.shape[:2] == states.shape[:2]

        # get the policy
        logits = self.pi(torch.cat([states, latent_skills], dim=-1)) * self.config.temp
        logits = self.stackActions(logits)
        assert logits.shape[:-1] == actions.shape

        return logits, self.monitor(states, actions, probs=False, pad_mask=dones), self.chooseSkill(states[:,0], logits=True), self.optForward(states)


    def optForward(self, states):
        assert states.dim() in [2, 3]
        assert states.shape[-1] == self.config.state_size

        return self.stackActions(self.opter(states))*self.config.temp


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