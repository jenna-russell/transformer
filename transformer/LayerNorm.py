import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(shape))
        self.b_2 = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        # LayerNorm(x + Sublayer(x))
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
