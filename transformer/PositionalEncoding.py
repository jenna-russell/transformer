import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.positional_encodings = torch.zeros(max_len, d_model)
        self.position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        self.positional_encodings[:, 0::2] = torch.sin(self.position * div_term)
        self.positional_encodings[:, 1::2] = torch.cos(self.position * div_term)

        self.positional_encodings = self.positional_encodings.unsqueeze(0)

        self.register_buffer("pe", self.positional_encodings)

    def forward(self, x):
        x = x + self.positional_encodings[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
