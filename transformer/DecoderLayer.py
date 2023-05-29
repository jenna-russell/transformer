from torch import nn

from transformer.SubLayerConnector import SubLayerConnector
from transformer.helperFunctions import clone_layer


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, multi_head_attn, ffn, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attn
        self.multihead_attention = multi_head_attn
        self.feed_forward_network = ffn
        self.sublayer = clone_layer(SubLayerConnector(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, source_mask, target_mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.multihead_attention(x, memory, memory, source_mask))
        return self.sublayer[2](x, self.feed_forward_network)