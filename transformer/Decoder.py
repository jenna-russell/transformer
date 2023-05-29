from torch import nn
from transformer.helperFunctions import clone_layer


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clone_layer(layer, n)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)