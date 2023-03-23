
import torch
from torch import nn
from torch.nn import functional as F

from model_utils import SkipNet, MobileNet


class QuantumPolicy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.delta = nn.Sequential(
            MobileNet(self.config.state_size),
            SkipNet(
                in_dim = self.config.state_size,
                h_dim = self.config.hidden_dim,
                out_dim = self.config.num_pi,
                n_layers = self.config.num_layers,
                dropout = self.config.dropout
            )
        )

        self.pi = nn.ModuleList([
            nn.Sequential(
                MobileNet(self.config.state_size),
                SkipNet(
                    in_dim = self.config.state_size,
                    h_dim = self.config.hidden_dim,
                    out_dim = self.config.action_dim*self.config.action_size,
                    n_layers = self.config.num_layers,
                    dropout = self.config.dropout
                )
        ) for _ in range(self.config.num_pi)])

        self.state_encoder = nn.Sequential(
            MobileNet(self.config.state_size),
            nn.Linear(self.config.state_size, self.config.latent_dim),
            nn.Softmax(dim=-1)
        )
        self.skill_encoder = nn.Sequential(
            nn.Linear(self.config.num_pi, self.config.latent_dim, bias=False)
        )

    def setSkill(self, skills):
        assert skills.dim() == 2
        assert skills.shape[-1] == self.config.num_pi

        self.curr_skill = skills.clone()
        self.n_curr = skills.shape[0]

        self.state_history = None
        self.action_history = None


    def getSkill(self):
        return self.curr_skill.clone().detach()


    def chooseSkill(self, states, logits=False):

        out = self.delta(states) * self.config.pred_temp
        p = torch.softmax(out, dim=-1)

        if logits:
            return out
        return p
    

    def setChooseSkill(self, states):
        skill = self.chooseSkill(states)
        self.setSkill(skill)
        return skill


    def policy(self, states, stochastic=True, action_override=None):
        assert self.curr_skill is not None
        assert states.shape[0] == self.n_curr

        if self.state_history is None:
            self.state_history = states.unsqueeze(1)
        else:
            self.state_history = torch.cat((self.state_history, states.unsqueeze(1)), dim=1)

        # get the policy
        pi = self.piForward(states)
        logits = self.mergePi(pi, self.curr_skill)

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

        return actions


    def forward(self, x):
        states, seed_states = x
        
        pi_logits = self.piForward(states)
        skill_logits = self.chooseSkill(seed_states, logits=True)
        
        pi_policy =  self.mergePi(pi_logits, torch.softmax(self.config.grad_temp * skill_logits, dim=-1).detach())
        skill_policy = self.mergePi(pi_logits.detach(), torch.softmax(skill_logits, dim=-1))

        enc_logits = skill_logits
        if not self.config.diff_delta:
            enc_logits = enc_logits.detach()

        skill_encs = self.skill_encoder(torch.softmax(enc_logits, dim=-1))
        skill_encs = F.normalize(skill_encs, p=1, dim=-1)

        state_encs = self.state_encoder(states)
        state_encs = F.normalize(state_encs, p=1, dim=-1)

        assert skill_encs.shape == state_encs.shape
        enc_outs = 5*((state_encs @ skill_encs.T)[:self.config.batch_keep,:self.config.batch_keep] - 1)

        return pi_policy, skill_policy, enc_outs, pi_logits, skill_logits


    def piForward(self, states):

        outs = [self.stackActions(self.pi[i](states)) for i in range(self.config.num_pi)]
        outs = torch.stack(outs, dim=-3) * self.config.pred_temp

        assert outs.shape[0] == states.shape[0]
        assert outs.shape[-3] == self.config.num_pi
        assert outs.shape[-2] == self.config.action_dim
        assert outs.shape[-1] == self.config.action_size

        return outs
    

    def mergePi(self, pi, skill):
        batched = skill.dim() == 2
        if not batched:
            skill = skill.unsqueeze(-1)
            pi = pi.unsqueeze(0)

        assert skill.shape[-1] == self.config.num_pi
        assert pi.shape[-3] == self.config.num_pi
        assert pi.shape[-2] == self.config.action_dim
        assert pi.shape[-1] == self.config.action_size

        if pi.dim() == 5:
            skill = skill.unsqueeze(1)
        skill = skill.unsqueeze(-1).unsqueeze(-1)
        assert skill.dim() == pi.dim()

        out = torch.sum(skill * pi, dim=-3)

        if not batched:
            out = out.squeeze(0)

        return out


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
