"""
Decoder unit tests.

Note: Blackbox testing of the decoder parameters.
"""
import unittest

import torch

from lib.nn import SeqDecoder
from tests.lib.utils import kwargs_product
from tests.lib.utils import tensor
from tests.lib.utils import random_args


class TestSeqDecoder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args().items():
            setattr(self, key, value)

        # Constant randomly generated tensors
        self.encoder_outputs = tensor(
            self.input_seq_len,
            self.batch_size,
            self.rnn_size,
            max_=self.input_vocab_size,
            type_=torch.FloatTensor)
        self.hidden = tensor(self.n_layers, self.batch_size, self.rnn_size, type_=torch.FloatTensor)

    def _decoders(self,
                  rnn_cell=['gru', 'lstm'],
                  embedding_dropout=[0, .5],
                  rnn_dropout=[0, .5],
                  use_attention=[True, False],
                  scheduled_sampling=[True, False]):
        """
        Args:
            contant arguments that are not changed
        Returns:
            generator for a list of all possible decoders with `possible_params`
        """
        possible_params = {}
        if rnn_cell:
            possible_params['rnn_cell'] = rnn_cell
        if embedding_dropout:
            possible_params['embedding_dropout'] = embedding_dropout
        if rnn_dropout:
            possible_params['rnn_dropout'] = rnn_dropout
        if use_attention:
            possible_params['use_attention'] = use_attention
        if scheduled_sampling:
            possible_params['scheduled_sampling'] = scheduled_sampling
        for kwargs in kwargs_product(possible_params):
            decoder = SeqDecoder(
                self.output_vocab_size,
                embedding_size=self.embedding_size,
                rnn_size=self.rnn_size,
                n_layers=self.n_layers,
                **kwargs)
            for param in decoder.parameters():
                param.data.uniform_(-.1, .1)
            yield decoder, kwargs

    def test_init(self):
        SeqDecoder(
            self.output_vocab_size, embedding_size=self.embedding_size, rnn_size=self.rnn_size)

    def test_parameters(self):
        """
        Check the setting different parameters, changes the module settings in the encoder.
        """
        representations = set()
        for decoder in self._decoders():
            self.assertTrue(str(decoder) not in representations)
            representations.add(str(decoder))

    # TODO: Test if `max_length=none` explicitly. This is challenging because the SeqDecoder only
    # stops if it predicts EOS token; therefore, without training it may loop forever.
    # NOTE: `max_length=none` is tested in the integration test where we do train.

    def test_forward_max_len(self):
        # Args
        max_length = self.output_seq_len + 2  # Account for <s> and </s>
        encoder_hidden = self.hidden
        target_output = tensor(
            self.output_seq_len + 2, self.batch_size, max_=self.output_vocab_size)

        # Run
        for decoder, decoder_kwargs in self._decoders(rnn_dropout=None, embedding_dropout=None):
            # LSTM rnn cell has a different hidden state then GRU
            if decoder_kwargs['rnn_cell'] == 'lstm':
                encoder_hidden = tuple([encoder_hidden, encoder_hidden])

            decoder_outputs, decoder_hidden, attention_weights = decoder.forward(
                max_length, encoder_hidden, self.encoder_outputs, target_output)

            if decoder_kwargs['rnn_cell'] == 'lstm':
                decoder_hidden = decoder_hidden[0]

            # Check sizes
            self.assertEqual(decoder_hidden.size(), (self.n_layers, self.batch_size, self.rnn_size))
            self.assertEqual(decoder_outputs.size(), (max_length, self.batch_size,
                                                      self.output_vocab_size))

            if decoder_kwargs['use_attention']:
                self.assertEqual(attention_weights.size(), (max_length, self.batch_size,
                                                            self.input_seq_len))

            # Check types
            self.assertEqual(decoder_hidden.data.type(), 'torch.FloatTensor')
            self.assertEqual(decoder_outputs.data.type(), 'torch.FloatTensor')
            if decoder_kwargs['use_attention']:
                self.assertEqual(attention_weights.data.type(), 'torch.FloatTensor')

            if decoder_kwargs['rnn_cell'] == 'lstm':
                encoder_hidden = encoder_hidden[0]

    def test_forward_step(self):
        # Args
        decoder_hidden = self.hidden
        last_decoder_output = tensor(
            self.output_seq_len,
            self.batch_size,
            max_=self.output_vocab_size,
            type_=torch.LongTensor)

        # Run
        for decoder, kwargs in self._decoders(rnn_dropout=None, embedding_dropout=None):
            # LSTM requires a different output/input
            if kwargs['rnn_cell'] == 'lstm':
                decoder_hidden = tuple([decoder_hidden, decoder_hidden])

            predicted_softmax, decoder_hidden_new, attention = decoder.forward_step(
                last_decoder_output, decoder_hidden, self.encoder_outputs)

            if kwargs['rnn_cell'] == 'lstm':
                decoder_hidden_new = decoder_hidden_new[0]
                decoder_hidden = decoder_hidden[0]  # Revert tuple

            # Check sizes
            self.assertEqual(decoder_hidden_new.size(), (self.n_layers, self.batch_size,
                                                         self.rnn_size))
            if kwargs['use_attention']:
                self.assertEqual(attention.size(), (self.batch_size, self.output_seq_len,
                                                    self.input_seq_len))
            self.assertEqual(predicted_softmax.size(), (self.batch_size, self.output_seq_len,
                                                        self.output_vocab_size))

            # Check types
            self.assertEqual(decoder_hidden_new.data.type(), 'torch.FloatTensor')
            if kwargs['use_attention']:
                self.assertEqual(attention.data.type(), 'torch.FloatTensor')
            self.assertEqual(predicted_softmax.data.type(), 'torch.FloatTensor')
