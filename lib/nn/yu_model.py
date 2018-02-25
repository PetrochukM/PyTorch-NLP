import torch
import torch.nn as nn

from lib.text_encoders import PADDING_INDEX
from lib.configurable import configurable


class YuModel(nn.Module):

    @configurable
    def __init__(self,
                 relation_vocab_size,
                 relation_word_vocab_size,
                 text_vocab_size,
                 embedding_size=300,
                 hidden_size=200,
                 dropout_relation=0.0,
                 dropout_text=0.0,
                 n_layers_relation=1):
        super().__init__()

        self.relation_embedding = nn.Embedding(
            relation_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.relation_word_embedding = nn.Embedding(
            relation_word_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.relation_word_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers_relation,
            dropout=dropout_relation,
            bidirectional=True)
        self.relation_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers_relation,
            dropout=dropout_relation,
            bidirectional=True)

        self.text_embedding = nn.Embedding(
            text_vocab_size, embedding_size, padding_idx=PADDING_INDEX)
        self.text_embedding.weight.requires_grad = False
        self.text_rnn_layer_one = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_text,
            bidirectional=True)

        self.text_rnn_layer_two = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_text,
            bidirectional=True)

        self.distance = nn.CosineSimilarity(dim=1)

    def forward(self, text, relation, relation_word):
        """
        Args:
            text: (torch.LongTensor [text_len, batch_size])
            relation: (torch.LongTensor [relation_len, batch_size])
            relation_word: (torch.LongTensor [relation_word_len, batch_size])
        """
        batch_size = text.size()[1]

        text = self.text_embedding(text)
        output_layer_one, _ = self.text_rnn_layer_one(text)
        output_layer_two, hidden = self.text_rnn_layer_two(output_layer_one)
        text = output_layer_one + output_layer_two  # Shortcut connection w/ point wise summation
        output_layer_one = None  # Clear memory
        output_layer_two = None  # Clear Memory
        text = torch.max(text, dim=0)[0]

        relation_word = self.relation_word_embedding(relation_word)
        relation_word, _ = self.relation_word_rnn(relation_word)

        relation = self.relation_embedding(relation)
        relation, _ = self.relation_rnn(relation)

        # (seq_len, batch_size, hidden_size * 2)
        relation = torch.cat([relation, relation_word], dim=0)
        # (seq_len * 2, batch_size, hidden_size * 2)
        relation = torch.max(relation, dim=0)[0]
        # (batch_size, hidden_size * 2)

        # (batch_size)
        return self.distance(text, relation)
