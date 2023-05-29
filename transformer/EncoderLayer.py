import torch.nn as nn
from transformer.SubLayerConnector import SubLayerConnector
from transformer.helperFunctions import clone_layer


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, ffn, dropout):
        # remember to add the dropout
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attn
        self.feed_forward_network = ffn
        self.sublayer = clone_layer(SubLayerConnector(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward_network)
