import torch
import torch.nn as nn


def clone_layer(layer, n):
    return nn.ModuleList([layer for _ in range(n)])


def mask(size):
    # mask the subsequent positions so we only attend to what we've seen so far
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

