import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embeddings, tgt_embeddings, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embeddings = src_embeddings
        self.output_embeddings = tgt_embeddings
        self.generator = generator

    def forward(self, input, input_mask, output, output_mask):
        # one step forward is taken every time we encode and then decode
        return self.decode(self.encoder(input, input_mask), input_mask, output, output_mask)

    def encode(self, input, input_mask):
        # maps sequence of symbols to a sequence of continuous representations
        return self.encoder(self.input_embeddings(input), input_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # maps sequence of continuous representations to a sequence of symbols
        return self.decoder(self.output_embeddings(tgt), memory, src_mask, tgt_mask)
