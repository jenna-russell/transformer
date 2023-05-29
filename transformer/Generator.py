import torch.nn as nn
from torch import log_softmax


class Generator(nn.Module):
    # the linear and softmax steps
    def __init__(self, decoder_output, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(decoder_output, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
