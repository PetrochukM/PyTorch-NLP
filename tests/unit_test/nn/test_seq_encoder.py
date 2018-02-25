"""
Encoder unit tests.

Note: Blackbox testing of the encoder parameters.
"""
import unittest

import torch

from lib.nn import SeqEncoder
from tests.lib.utils import kwargs_product
from tests.lib.utils import tensor
from tests.lib.utils import random_args


class TestSeqEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args().items():
            setattr(self, key, value)

        # Constant randomly generated tensors
        self.input = tensor(
            self.input_seq_len, self.batch_size, max_=self.input_vocab_size, type_=torch.LongTensor)

    def _encoders(self):
        """
        Returns:
            (generator) generator for a list of possible encoders with `possible_kwargs`
        """
        possible_kwargs = {
            'rnn_cell': ['gru', 'lstm'],
            'embedding_dropout': [0, .5],
            'rnn_dropout': [0, .5],
            'n_layers': [1, 2],
            'bidirectional': [True, False],
            'freeze_embeddings': [True, False],
        }
        for kwargs in kwargs_product(possible_kwargs):
            decoder = SeqEncoder(
                self.input_vocab_size,
                embedding_size=self.embedding_size,
                rnn_size=self.rnn_size,
                **kwargs)
            for param in decoder.parameters():
                param.data.uniform_(-1, 1)
            yield decoder, kwargs

    def test_init(self):
        SeqEncoder(self.input_vocab_size)

    def test_parameters(self):
        """
        Check the setting different parameters, changes the module settings in the encoder.
        """
        representations = set()
        for encoder in self._encoders():
            print(str(encoder))
            self.assertTrue(str(encoder) not in representations)
            representations.add(str(encoder))

    def test_forward(self):
        # Args
        lengths = torch.LongTensor(self.batch_size).fill_(self.input_seq_len)

        # Run
        for encoder, encoder_kwargs in self._encoders():
            # LSTM rnn cell has a different hidden state then GRU
            encoder_output, encoder_hidden = encoder.forward(self.input, lengths)

            if encoder_kwargs['rnn_cell'] == 'lstm':
                encoder_hidden = encoder_hidden[0]

            # Check sizes
            self.assertEqual(encoder_hidden.size(), (encoder_kwargs['n_layers'], self.batch_size,
                                                     self.rnn_size))
            self.assertEqual(encoder_output.size(), (self.input_seq_len, self.batch_size,
                                                     self.rnn_size))

            # Check types
            self.assertEqual(encoder_hidden.data.type(), 'torch.FloatTensor')
            self.assertEqual(encoder_output.data.type(), 'torch.FloatTensor')
