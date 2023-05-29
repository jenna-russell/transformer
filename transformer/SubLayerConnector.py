import torch.nn as nn
from transformer.LayerNorm import LayerNorm


class SubLayerConnector(nn.Module):
    # employ residual connection after each of the sublayers
    def __init__(self, size, dropout):
        super(SubLayerConnector, self).__init__()
        self.norm = LayerNorm(size)
        # Dropout: the practice of disregarding certain nodes in a layer at random during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # normalize x and add dropout
        return x + self.dropout(sublayer(self.norm(x)))
