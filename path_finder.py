
import torch
from torch import nn
from torch.nn import functional as F

from configs import DefaultPathFinder
from model_utils import PositionalEncoding


class PathFinder(nn.Module):

    def __init__(self, config=DefaultPathFinder):
        super().__init__()
        self.config = config
        
        self.pos_enc = PositionalEncoding(self.config.h_dim, self.config.max_seq_len)
        
        self.target_head = nn.Sequential(
            nn.Linear(self.config.state_size, self.config.mid_dim),
            nn.ELU(),
            nn.Linear(self.config.mid_dim, self.config.h_dim-1),
        )
        self.head = nn.Sequential(
            nn.Linear(self.config.state_size, self.config.mid_dim),
            nn.ELU(),
            nn.Linear(self.config.mid_dim, self.config.h_dim),
        )
        self.tail = nn.Sequential(
            nn.Linear(self.config.h_dim, self.config.mid_dim),
            nn.ELU(),
            nn.Linear(self.config.mid_dim, self.config.state_size),
        )

        enc_layer = nn.TransformerEncoderLayer(self.config.h_dim, self.config.num_heads_encoding, dim_feedforward=self.config.dim_feedforward_encoding, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.num_layers_encoding)   
        
        finder_layer = nn.TransformerDecoderLayer(self.config.h_dim, self.config.num_heads, dim_feedforward=self.config.dim_feedforward, batch_first=True)
        self.finder = nn.TransformerDecoder(finder_layer, num_layers=self.config.num_layers)

        self.start_token = torch.tensor([1.0] + [0.0] * (self.config.h_dim-1))
        self.end_token = torch.tensor([0.0] * (self.config.h_dim-1) + [1.0])
        

    def encode(self, states):
        # states is (batch, seq_len, state_size)

        l = self.head(states)
        l = F.normalize(l, dim=-1)
        
        tok_start = self.start_token.to(states.device)
        tok_end = self.end_token.to(states.device)
        if states.dim() == 3:
            tok_start = torch.stack([self.start_token]*states.shape[0], dim=0).to(states.device)
            tok_end = torch.stack([self.end_token]*states.shape[0], dim=0).to(states.device)
        
        l = torch.cat((tok_start.unsqueeze(-2), l, tok_end.unsqueeze(-2)), dim=-2)
        
        l = self.pos_enc(l)

        out = self.encoder(l)

        out = torch.mean(out, -2)
        
        return F.normalize(out, dim=-1)


    def forward(self, x, target_traj):

        # get encoding of target
        target = target_traj[:, -1:, :]
        latent_target = self.target_head(target)
        latent_target = F.normalize(latent_target, dim=-1)
        latent_target = torch.cat([latent_target, torch.ones_like(latent_target[:, :, :1])], dim=-1)
    
        # get the encoding of the sequence
        enc = self.encode(target_traj)
        
        # cat the encoding and target to ge tmemory input
        memory = torch.cat([enc.unsqueeze(-2), latent_target], dim=-2)
        memory = self.pos_enc(memory)
    
        # store the first element of the sequence
        raw_start = x[:, 0:1, :]
    
        # get the encoding of the sequence
        latent_seq = self.head(x)
        latent_seq = F.normalize(latent_seq, dim=-1)
        
        # remove the last element of the sequence
        latent_tgt = latent_seq[:, :-1, :]
        # add positional encoding
        latent_tgt = self.pos_enc(latent_tgt)
        
        # get the temporal mask
        temporal_mask = torch.full((latent_tgt.shape[-2], latent_tgt.shape[-2]), float('-inf'), device=latent_tgt.device)
        for i in range(latent_tgt.shape[-2]):
            temporal_mask[i, :i+1] = 0

        # get the predictions
        latent_out = self.finder(latent_tgt, memory, tgt_mask=temporal_mask)
        latent_out = F.normalize(latent_out, dim=-1)

        # get the final state predictions
        raw_out = self.tail(latent_out)
        
        # add the first element to the predictions to make langths match
        out = torch.cat([raw_start, raw_out], dim=-2)
        
        # get predictions about when we reach the target
        pred_end = torch.zeros_like(raw_out[:,:,0])

        return out, pred_end


    def deadReckon(self, x):
        
        # get encoding of target
        target = x[:, -1:, :]
        latent_target = self.head(target)
        latent_target = F.normalize(latent_target, dim=-1)
    
        # get the encoding of the sequence
        enc = self.encode(x)
        memory = torch.cat([enc.unsqueeze(-2), latent_target], dim=-2)
        memory = self.pos_enc(memory)
    
        # get the encoding of the sequence
        latent_seq = self.head(x[:,:1,:])
        
        # add positional encoding
        input_seq = self.pos_enc(latent_seq)
        
        for i in range(x.shape[-2]-1):
            
            # get the temporal mask
            temporal_mask = torch.full((input_seq.shape[-2], input_seq.shape[-2]), float('-inf'), device=input_seq.device)
            for i in range(input_seq.shape[-2]):
                temporal_mask[i, :i+1] = 0

            # get the predictions
            out = self.finder(input_seq, memory, tgt_mask=temporal_mask)
            out = F.normalize(out, dim=-1)
            
            # add the first element to the predictions to make langths match
            out = torch.cat([latent_seq[:, 0:1, :], out], dim=-2)

            # get the final state predictions
            pred_seq = self.tail(out)
        
        # get predictions about when we reach the target
        pred_end = torch.zeros_like(torch.sum(latent_target * out, dim=-1))

        return pred_seq