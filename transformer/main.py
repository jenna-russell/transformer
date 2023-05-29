# This is a sample Python script.
import copy

import torch
from torch import nn

from transformer.Decoder import Decoder
from transformer.DecoderLayer import DecoderLayer
from transformer.Embeddings import Embeddings
from transformer.Encoder import Encoder
from transformer.EncoderLayer import EncoderLayer
from transformer.Generator import Generator
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.PositionWiseFeedForwardNetwork import PositionWiseFeedForwardNetwork
from transformer.PositionalEncoding import PositionalEncoding
from transformer.Transformer import Transformer
from transformer.helperFunctions import mask


def make_model(source_vocab, target_vocab, h=8, n=6, d_model=512, d_ff=2048, dropout=0.1):
    attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForwardNetwork(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    encoder_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout)
    encoder = Encoder(encoder_layer, n)
    encoder_embeddings = nn.Sequential(Embeddings(d_model, source_vocab), copy.deepcopy(position))

    decoder_layer = DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout)
    decoder = Decoder(decoder_layer, n)
    decoder_embeddings = nn.Sequential(Embeddings(d_model, target_vocab))

    generator = Generator(d_model, target_vocab)

    transformer_model = Transformer(encoder, decoder, encoder_embeddings, decoder_embeddings, generator)

    # im very confused by this part of the implementation
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for i in range(10):
        inference_test()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_tests()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
