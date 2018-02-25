import torch.nn as nn
import torch.nn.functional as F

from lib.configurable import configurable
from lib.nn.seq_encoder import SeqEncoder


class SeqToLabel(nn.Module):

    @configurable
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 freeze_embeddings=False,
                 bidirectional=False,
                 embedding_size=128,
                 rnn_size=128,
                 rnn_cell='lstm',
                 rnn_layers=1,
                 decode_dropout=0.0,
                 rnn_variational_dropout=0.0,
                 rnn_dropout=0.0,
                 embedding_dropout=0.0):
        super().__init__()

        self.encoder = SeqEncoder(
            vocab_size=input_vocab_size,
            embedding_size=embedding_size,
            rnn_size=rnn_size,
            embedding_dropout=embedding_dropout,
            rnn_variational_dropout=rnn_variational_dropout,
            n_layers=rnn_layers,
            rnn_cell=rnn_cell,
            bidirectional=bidirectional,
            rnn_dropout=rnn_dropout,
            freeze_embeddings=freeze_embeddings)

        self.out = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),  # can apply batch norm after this - add later
            nn.BatchNorm1d(rnn_size),
            nn.ReLU(),
            nn.Dropout(p=decode_dropout),
            nn.Linear(rnn_size, output_vocab_size))

    def forward(self, text, mask=None):
        _, hidden = self.encoder(text)
        if self.encoder.rnn_cell == nn.LSTM:
            hidden = hidden[0]

        output = self.out(hidden[-1])
        if mask is not None:
            output = output * mask
        scores = F.log_softmax(output, dim=1)
        return scores
