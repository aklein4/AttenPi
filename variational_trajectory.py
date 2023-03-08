
import torch
from torch import nn
from torch.nn import functional as F

from configs import DefaultVariationalTrajectory
from model_utils import PositionalEncoding


class VariationalTrajectory(nn.Module):

    def __init__(self, config=DefaultVariationalTrajectory, dead_reckon=True):
        super().__init__()
        self.config = config
        
        self.dead_reckon = dead_reckon
        
        self.enc_pos = PositionalEncoding(self.config.l_dim*2, self.config.max_seq_len)
        self.dec_pos = PositionalEncoding(self.config.l_dim, self.config.max_seq_len)
        
        self.encoder_head = nn.Sequential(
            nn.Linear(self.config.state_size, self.config.mid_dim),
            nn.ELU(),
            nn.Linear(self.config.mid_dim, self.config.l_dim*2),
        )
        
        self.decoder_head = nn.Sequential(
            nn.Linear(self.config.state_size, self.config.mid_dim),
            nn.ELU(),
            nn.Linear(self.config.mid_dim, self.config.l_dim),
        )
        
        self.decoder_tail = nn.Sequential(
            nn.Linear(self.config.l_dim, self.config.mid_dim),
            nn.ELU(),
            nn.Linear(self.config.mid_dim, self.config.state_size),
        )

        enc_layer = nn.TransformerEncoderLayer(self.config.l_dim*2, self.config.num_heads_encoding, dim_feedforward=self.config.dim_feedforward_encoding, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.num_layers_encoding)   
        
        decoder_layer = nn.TransformerDecoderLayer(self.config.l_dim, self.config.num_heads, dim_feedforward=self.config.dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_layers)

        self.start_token = torch.tensor([1.0] + [0.0] * (2*self.config.l_dim - 1))
        self.end_token = torch.tensor([0.0] * (2*self.config.l_dim - 1) + [1.0])
        

    def encode(self, states):
        
        batched = True
        if states.dim() == 2:
            batched = False
            states = states.unsqueeze(0)

        l = self.encoder_head(states)
        
        tok_start = self.start_token.to(states.device)
        tok_end = self.end_token.to(states.device)
        if states.dim() == 3:
            tok_start = torch.stack([self.start_token]*states.shape[0], dim=0).to(states.device)
            tok_end = torch.stack([self.end_token]*states.shape[0], dim=0).to(states.device)
        
        l = torch.cat((tok_start.unsqueeze(-2), l, tok_end.unsqueeze(-2)), dim=-2)
        
        l = self.enc_pos(l)

        out = self.encoder(l)

        out = torch.mean(out, -2).unsqueeze(-2) 
        
        mus, sigmas = out[:,:,:self.config.l_dim], torch.exp(out[:,:,self.config.l_dim:])
        
        if not batched:
            mus = mus.squeeze(0)
            sigmas = sigmas.squeeze(0)
        
        return mus, sigmas


    def forward(self, x, variance=True):
    
        # get the encoding of the sequence
        mus, sigmas = self.encode(x)
        
        # use latent sampling to 
        memory = mus
        if variance:
            memory += sigmas * torch.randn_like(sigmas)
    
        if self.dead_reckon:
            return self.decode(memory, x[:, 0], x.shape[-2]), mus, sigmas
    
        # store the first element of the sequence
        raw_start = x[:, 0:1, :]
    
        # get the encoding of the sequence
        latent_tgt = self.decoder_head(x)
        
        # remove the last element of the sequence
        latent_tgt = latent_tgt[:, :-1, :]
        # add positional encoding
        latent_tgt = self.dec_pos(latent_tgt)
        
        # get the temporal mask
        temporal_mask = torch.full((latent_tgt.shape[-2], latent_tgt.shape[-2]), float('-inf'), device=latent_tgt.device)
        for i in range(latent_tgt.shape[-2]):
            temporal_mask[i, :i+1] = 0

        # get the predictions
        latent_out = self.decoder(latent_tgt, memory, tgt_mask=temporal_mask)

        # get the final state predictions
        raw_out = self.decoder_tail(latent_out)
        
        # add the first element to the predictions to make langths match
        out = torch.cat([raw_start, raw_out], dim=-2)

        return out, mus, sigmas


    def decode(self, encoding, start, length):
        
        if isinstance(encoding, tuple):
            encoding = encoding[0]
        
        batched = True
        if encoding.dim() == 2:
            batched = False
            encoding = encoding.unsqueeze(0)
        
        # use latent sampling to 
        memory = encoding

        assert start.dim() == 1 and not batched or start.dim() == 2 and batched
        if not batched:
            start = start.unsqueeze(0)
    
        start = start.unsqueeze(-2)
        seq = start
        
        out = seq
        
        torch.no_grad()
        for i in range(length-1):
            if i+1 == length-1:
                torch.enable_grad()
        
            # get the encoding of the sequence
            latent_tgt = self.decoder_head(seq)
        
            # add positional encoding
            latent_tgt = self.dec_pos(latent_tgt)
            
            # get the temporal mask
            temporal_mask = torch.full((latent_tgt.shape[-2], latent_tgt.shape[-2]), float('-inf'), device=latent_tgt.device)
            for i in range(latent_tgt.shape[-2]):
                temporal_mask[i, :i+1] = 0

            # get the predictions
            latent_out = self.decoder(latent_tgt, memory, tgt_mask=temporal_mask)

            # get the final state predictions
            raw_out = self.decoder_tail(latent_out)
            
            # add the first element to the predictions to make langths match
            out = torch.cat([start, raw_out], dim=-2)

            seq = out.detach()

        if not batched:
            out = seq.squeeze(0)

        return out