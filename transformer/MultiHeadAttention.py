import math

import torch
from torch import nn

from transformer.helperFunctions import clone_layer


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        # why is this 4
        self.linear_layers = clone_layer(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # do linear projections in batch
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear_layers, (query, key, value))
        ]

        # 512 * 512
        # b, t, k - training data

        # view(b, t, h, d_k)
        # b = batches
        # t = sequence -1
        # h = 8 (heads)
        # d_k = 64 (512 / 8)

        # head representation: (b, h, t, d_k)

        # nn.linear()

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # attn = (b, h, t, t)
        # x = (b , h, t, d_k)

        x = (x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k))

        # x = (b, t, h, d_k) -> (b, -1, 512)
        #t is the sequence length

        del query
        del key
        del value
        return self.linear_layers[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = key.size(-1)
    # d_k = 64

    # queries : (b, h, t, d_k)
    # keys : (b, h, d_k, t)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # scores = (b, h, t, t)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = scores.softmax(-1)
    # attn normalizes the scores
    if dropout is not None:
        attn = dropout(attn)
    x = torch.matmul(attn, value)
    # values : (b, h, t, d_k)
    # attn= (b, h, t, t)
    # x = (b , h, t, d_k)

    return x, attn
