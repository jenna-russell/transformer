import math
from torch import nn


class Embeddings(nn.Module):
    def __init__(self, d_model,
                 vocab):
        super(Embeddings, self).__init__()
        self.learned_embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.learned_embedding(x.long()) * math.sqrt(self.d_model)
        return embed
