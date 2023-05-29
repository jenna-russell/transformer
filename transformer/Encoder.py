import torch.nn as nn
import helperFunctions
from transformer.LayerNorm import LayerNorm


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = helperFunctions.clone_layer(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            # compute layer on x and update value of x
            x = layer(x, mask)
        # normalize
        return self.norm(x)
