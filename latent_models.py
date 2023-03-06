
import torch
from torch import nn
from torch.nn import functional as F

from configs import DefaultLatentTrajectory
from utils import PositionalEncoding

import numpy as np


class LatentTrajectory(nn.Module):

    def __init__(self, config=DefaultLatentTrajectory):
        super().__init__()
        self.config = config

        self.encoder_mouth = nn.Linear(self.config.state_size, self.config.h_dim)
        
        self.decoder_mouth = nn.Linear(self.config.state_size, self.config.h_dim)
        self.decoder_tail  = nn.Linear(self.config.h_dim, self.config.state_size)

        self.pos_enc = PositionalEncoding(self.config.h_dim, self.config.max_seq_len)

        enc_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_encoding_heads, dim_feedforward=self.config.dim_encoding_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.num_encoding_layers)

        dec_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_decoding_heads, dim_feedforward=self.config.dim_decoding_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.config.num_decoding_layers)

        self.start_token = torch.tensor([1.0] + [0.0] * (self.config.h_dim-1))
        self.end_token = torch.tensor([0.0] * (self.config.h_dim-1) + [1.0])
        


    def encode(self, states):
        # states is (batch, seq_len, state_size)

        l = self.encoder_mouth(states)
        
        tok_start = self.start_token.to(states.device)
        tok_end = self.end_token.to(states.device)
        if states.dim() == 3:
            tok_start = torch.stack([self.start_token]*states.shape[0], dim=0).to(states.device)
            tok_end = torch.stack([self.end_token]*states.shape[0], dim=0).to(states.device)
        
        l = torch.cat((tok_start.unsqueeze(-2), l, tok_end.unsqueeze(-2)), dim=-2)
        
        l = self.pos_enc(l)

        out = self.encoder(l)

        out = torch.mean(out, -2)
        
        return F.normalize(out, dim=-1) if self.config.norm_l else out


    def decode(self, encoding, states):
        if states.dim() == 3:
            assert states.shape[0] == encoding.shape[0]
        elif states.dim() == 2:
            assert encoding.dim() == 1

        encoding = encoding.unsqueeze(-2)

        l = self.decoder_mouth(states)
        
        tok_start = self.start_token
        if states.dim() == 3:
            tok_start = torch.stack([self.start_token]*states.shape[0], dim=0)
        l = torch.cat((tok_start.unsqueeze(-2).to(states.device), l), dim=-2)

        l = self.pos_enc(l)

        tgt_mask = torch.full((l.shape[-2], l.shape[-2]), float('-inf'))
        for i in range(l.shape[-2]):
            tgt_mask[i, :i+1] = 0

        out = self.decoder(l, encoding, tgt_mask=tgt_mask.to(states.device))
        out = self.decoder_tail(out)
        
        if states.dim() == 3:
            out = out[:, :-1, :]
            # out[:,1:] += states[:,1:]
        else:
            out = out[:-1, :]
            # out[1:] += states[:-1]

        return out


class LatentPolicy(nn.Module):
    def __init__(self, config=DefaultLatentTrajectory):
        super().__init__()
        self.config = config
        
        self.decoder_mouth = nn.Linear(self.config.state_size, self.config.h_dim)
        self.decoder_tail  = nn.Linear(self.config.h_dim, self.config.action_size)

        self.pos_enc = PositionalEncoding(self.config.h_dim, self.config.max_seq_len)

        dec_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_decoding_heads, dim_feedforward=self.config.dim_decoding_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.config.num_decoding_layers)

        self.start_token = torch.tensor([1.0] + [0.0] * (self.config.h_dim-1))

    


def main():
    pass

if __name__ == '__main__':
    main()