
import torch
from torch import nn
from torch.nn import functional as F

from configs import DefaultLatentPolicy

import numpy as np


class LatentPolicy(nn.Module):

    def __init__(self, config=DefaultLatentPolicy):
        super().__init__()
        self.config = config

        self.encoder_mouth = nn.Linear(self.config.state_size + self.config.action_size, self.config.h_dim)
        
        self.decoder_mouth = nn.Linear(self.config.state_size, self.config.h_dim)
        self.decoder_tail  = nn.Linear(self.config.h_dim, self.config.action_size)

        enc_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_encoding_heads, dim_feedforward=self.config.dim_encoding_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.num_encoding_layers)

        dec_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_decoding_heads, dim_feedforward=self.config.dim_decoding_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.config.num_decoding_layers)

        self.inference_policy = None


    def encodePolicy(self, states, actions):


        # x is state-actions tuple
        l = torch.cat((states, actions), dim=-1)

        l = self.encoder_mouth(l)

        out = self.encoder(l)

        out = torch.mean(out, -2)
        
        return F.normalize(out, dim=-1) if self.config.norm_l else out


    def decodePolicy(self, policy, states, probs=False):
        if states.dim() == 3:
            assert states.shape[0] == policy.shape[0]
        elif states.dim() == 2:
            assert policy.dim() == 1

        policy = policy.unsqueeze(-2)

        h_actions = self.decoder_mouth(states)

        tgt_mask = torch.full((states.shape[-2], states.shape[-2]), float('-inf'))
        tgt_mask.fill_diagonal_(0)

        out = self.decoder(h_actions, policy, tgt_mask=tgt_mask)

        pred_actions = self.decoder_tail(out)

        return F.softmax(pred_actions, dim=-1) if probs else pred_actions


    def setInferencePolicy(self, policy):
        self.inference_policy = policy
    

    def sampleAction(self, state):
        assert self.inference_policy is not None

        probs = self.decodePolicy(self.inference_policy, state.unsqueeze(0), probs=True)[0].detach().cpu().numpy()
        return np.random.choice(range(self.config.action_size), p=probs)


    def greedyAction(self, state):
        assert self.inference_policy is not None

        return torch.argmax(self.decodePolicy(self.inference_policy, state.unsqueeze(0))[0], dim=-1).item()


def main():
    pass

if __name__ == '__main__':
    main()