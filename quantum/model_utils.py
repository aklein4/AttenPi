
import torch
import torch.nn as nn


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
        if X.dim() == 3:
            X = X + self.P[:, :X.shape[1], :].to(X.device)
        else:
            X = X + self.P[0, :X.shape[0], :].to(X.device)
        return X
    

class SkipNet(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout):
        super().__init__()

        self.in_layer = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Dropout(dropout),
            nn.ELU(),
        )
        
        self.mid_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.Dropout(dropout),
                    nn.ELU(),
                )
            for _ in range(n_layers)]
        )

        self.out_layer = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, out_dim)
        )

        self.net = nn.Sequential(
            self.in_layer,
            self.mid_layers,
            self.out_layer
        )


    def forward(self, x):
        return self.net(x)
