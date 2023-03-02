
import torch
from torch import nn


class StateActionEncoderDecoder(nn.Module):

    def __init__(self, n_state, n_actions, n_hidden, n_encode, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_state + n_actions, n_hidden),
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(n_hidden, n_encode)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_encode, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_state + n_actions),
            nn.Tanh()
        )


    def encode(self, x):
        enc = self.encoder(x)
        return nn.functional.normalize(enc, dim=-1)

    def decode(self, x):
        return 2*self.decoder(x)


    def forward(self, x):
        return self.decode(self.encode(x))


# https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, max_len):
        super().__init__()

        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X


class TauEncoderDecoder(nn.Module):

    def __init__(self, input_dim, h_dim, max_len, num_heads, num_layers, dim_feedforward):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.max_len = max_len

        self.upscaler = nn.Linear(input_dim, h_dim)
        self.downscaler = nn.Linear(h_dim, input_dim)

        self.pos_enc = PositionalEncoding(h_dim, self.max_len)

        enc_layer = nn.TransformerEncoderLayer(h_dim, num_heads, batch_first=True, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(h_dim, num_heads, batch_first=True, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)


    def encode(self, x):
        x = self.upscaler(x)

        x = x + self.pos_enc.forward(x)
        enc = self.encoder(x)[:,0,:]

        return nn.functional.normalize(enc, dim=-1)


    def decode(self, x, stop_thresh=0.8):

        mem = torch.unsqueeze(x, 1)
        
        start_token = torch.zeros_like(mem)
        start_token[:,:,0] = 1

        end_token = torch.zeros((start_token.shape[-1],), dtype=torch.float32).to(x.device)
        end_token[-1] = 1

        tgt = start_token
        out = None

        while True:

            tgt_mask = torch.full((tgt.shape[1], tgt.shape[1]), 1, dtype=torch.bool).to(x.device)
            for i in range(tgt.shape[1]):
                tgt_mask[i, i:] = 0

            out = self.decoder(tgt, mem, tgt_mask=tgt_mask)
            tgt = torch.cat((start_token, out), dim=1)

            if out[:,-1] @ end_token >= 100 or tgt.shape[1] >= self.max_len:
                break

        out = self.downscaler(out)
        return torch.nn.functional.normalize(out, dim=-1)


    def forward(self, x):

        memory = self.encode(x).unsqueeze(1)

        tgt = self.upscaler(x)
        start_token = torch.zeros_like(tgt[:,0,:]).unsqueeze(1)
        start_token[:,:,0] = 1
        tgt = torch.cat((start_token, tgt), dim=1)

        tgt_mask = torch.full((tgt.shape[1], tgt.shape[1]), 1, dtype=torch.bool).to(x.device)
        for i in range(tgt.shape[1]):
            tgt_mask[i, i:] = 0

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        out = self.downscaler(out)

        return torch.nn.functional.normalize(out, dim=-1)
        
    


def main():
    pass

if __name__ == '__main__':
    main()