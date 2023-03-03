
import torch
from torch import nn

from utils import PositionalEncoding
from configs import DefaultTrajectory2Policy


class Trajectory2Policy(nn.Module):

    def __init__(self, config=DefaultTrajectory2Policy):
        super().__init__()
        self.config = config

        self.encoder_mouth = nn.Linear(self.config.state_size + self.config.action_size, self.config.h_dim)
        self.decoder_mouth = nn.Linear(self.config.state_size + self.config.action_size, self.config.h_dim)

        self.embed2action = nn.Linear(self.config.h_dim, self.config.action_size)

        self.pos_enc = PositionalEncoding(self.config.h_dim, self.config.seq_len)

        enc_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_encoding_heads, dim_feedforward=self.config.dim_encoding_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.num_encoding_layers)

        dec_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_decoding_heads, dim_feedforward=self.config.dim_decoding_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.config.num_decoding_layers)


    def encodePolicy(self, states, actions):

        # x is state-actions tuple
        l = torch.cat((states, actions), dim=-1)

        l = self.encoder_mouth(l)

        if self.config.temporal_encoding:
            l = self.pos_enc.forward(l)

        out = self.encoder(l)

        out = torch.mean(out, -2).unsqueeze(-2) / self.config.h_dim**0.5
        
        return out


    def decodePolicy(self, policy, states, actions, probs=False):
        assert states.shape[-2] == actions.shape[-2] and states.shape[-2] <= self.config.seq_len

        actions = torch.roll(actions, 1, -2)        
        if actions.dim() == 3:
            actions[:,:,:] = 0
        else:
            actions[:,:] = 0

        l = torch.cat((states, actions), dim=-1)
        l = self.decoder_mouth(l)

        if self.config.temporal_decoding:
            l = self.pos_enc.forward(l)

        state_mask = torch.full([l.shape[-2], l.shape[-2]], float("-inf"), dtype=torch.float).to(policy.device)
        for i in range(l.shape[-2]):
            if self.config.remember_past:
                state_mask[i, :i] = 0
            else:
                state_mask[i, i] = 0

        out = self.decoder(l, policy, tgt_mask=state_mask)

        policy_actions = self.embed2action(out)

        return torch.nn.functional.softmax(policy_actions, dim=-1) if probs else policy_actions


def main():
    pass

if __name__ == '__main__':
    main()