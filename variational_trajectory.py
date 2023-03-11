
import torch
from torch import nn
from torch.nn import functional as F

from configs import DefaultVariationalTrajectory
from model_utils import PositionalEncoding


class VariationalTrajectory(nn.Module):

    def __init__(self, config=DefaultVariationalTrajectory):
        super().__init__()
        self.config = config
        
        self.enc_pos = PositionalEncoding(self.config.l_dim*2, self.config.max_seq_len)
        self.dec_pos = PositionalEncoding(self.config.l_dim, self.config.max_seq_len)
        
        self.encoder_head = nn.Sequential(
            nn.Linear(self.config.state_size + self.config.action_size + 2, self.config.l_dim*2),
        )
        
        self.decoder_head = nn.Sequential(
            nn.Linear(self.config.state_size, self.config.l_dim),
        )

        embeds = torch.zeros((self.config.l_dim, self.config.action_size)).unsqueeze(0).unsqueeze(0)
        nn.init.xavier_uniform_(embeds)
        self.action_embeddings = nn.Parameter(embeds, requires_grad=True)

        enc_layer = nn.TransformerEncoderLayer(self.config.l_dim*2, self.config.num_heads_encoding, dim_feedforward=self.config.dim_feedforward_encoding, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.config.num_layers_encoding)   
        
        decoder_layer = nn.TransformerDecoderLayer(self.config.l_dim, self.config.num_heads, dim_feedforward=self.config.dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_layers)
        

    def encode(self, x):
        
        states, actions = x
        
        # encode the states into a latent space
        
        assert states.dim() == actions.dim()
        
        batched = True
        if states.dim() == 2:
            batched = False
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
        
        action_part = torch.zeros((actions.shape[0], actions.shape[1], self.config.action_size+2), device=states.device)
        action_part[:,:,:4][actions] = 1.0
        
        l = torch.cat((states, action_part), dim=-1)
        
        tok_start = torch.zeros_like(l[:,:1,:])
        tok_start[:,:,-2] = 1.0
        
        tok_end = torch.zeros_like(l[:,:1,:])
        tok_end[:,:,-1] = 1.0
        
        l = torch.cat((tok_start, l, tok_end), dim=-2)
    
        l = self.encoder_head(l)
        l = self.enc_pos(l)

        out = self.encoder(l)

        out = torch.mean(out, -2).unsqueeze(-2) 
        
        mus, sigmas = out[:,:,:self.config.l_dim], torch.exp(out[:,:,self.config.l_dim:])
        
        if not batched:
            mus = mus.squeeze(0)
            sigmas = sigmas.squeeze(0)
        
        return mus, sigmas


    def forward(self, x, variance=True):
    
        x, actions = x
    
        # get 
        batched=True
        if x.dim() == 2:
            batched = False
            x = x.unsqueeze(0)
            actions = actions.unsqueeze(0)
    
        # get the encoding of the sequence
        mus, sigmas = self.encode((x, actions))
        
        # use latent sampling to 
        memory = mus
        if variance:
            memory += sigmas * torch.randn_like(sigmas)
    
        # get the encoding of the sequence
        latent_tgt = self.decoder_head(x)
        
        # add positional encoding
        latent_tgt = self.dec_pos(latent_tgt)
        
        # get the temporal mask
        temporal_mask = torch.full((latent_tgt.shape[-2], latent_tgt.shape[-2]), float('-inf'), device=latent_tgt.device)
        for i in range(latent_tgt.shape[-2]):
            temporal_mask[i, :i+1] = 0

        # get the predictions
        latent_out = self.decoder(latent_tgt, memory, tgt_mask=temporal_mask)

        # convert to actions
        action_logits = latent_out.unsqueeze(-1) * F.normalize(self.action_embeddings, dim=-2)
        action_logits = torch.sum(action_logits, dim=-2)
        
        if not batched:
            action_logits = action_logits.squeeze(0)
            mus = mus.squeeze(0)
            sigmas = sigmas.squeeze(0)

        action_probs = torch.softmax(action_logits, -1)

        return action_probs, mus, sigmas


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