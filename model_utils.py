
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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
                    nn.Linear(2*h_dim, h_dim),
                    nn.Dropout(dropout),
                    nn.ELU(),
                )
            for _ in range(n_layers)]
        )

        self.out_layer = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.ELU(dropout),
            nn.Linear(h_dim, out_dim)
        )


    def forward(self, x):
        h = self.in_layer(x)
        prev = h
        for layer in self.mid_layers:
            temp = h
            h = layer(torch.cat([h, prev], dim=-1))
            prev = temp

        return self.out_layer(torch.cat([h, prev], dim=-1))


MOBILE_PRETRAINED = "./local_data/pre-conv.pt"
class MobileNet(nn.Module):
    def __init__(self, out_features, load=None):
        super().__init__()

        self.model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.model.features[0][0] = nn.Conv2d(9, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, out_features, bias=True)

        if load is not None:
            self.model.load_state_dict(torch.load(load))
    
    def forward(self, x):
        return self.model(x)
