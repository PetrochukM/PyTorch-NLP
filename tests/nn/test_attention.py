"""
Attention unit tests.

Note: Blackbox testing of the attention.
"""
import random
import unittest

import torch

from lib.nn import Attention
from tests.nn.utils import kwargs_product
from tests.nn.utils import tensor


class TestAttention(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Some random small constants to set up the hidden state sizes
        self.batch_size = random.randint(1, 10)
        self.dimensions = random.randint(1, 10)
        self.output_seq_len = random.randint(1, 10)
        self.input_seq_len = random.randint(1, 10)

        # Constant randomly generated tensors
        self.input_ = tensor(
            self.batch_size, self.output_seq_len, self.dimensions, type_=torch.FloatTensor)
        self.context = tensor(
            self.batch_size, self.input_seq_len, self.dimensions, type_=torch.FloatTensor)

    def _attentions(self, attention_type=['general', 'dot']):
        """
        Generate all possible instantiations of `Attention` to test
        """
        possible_params = {}
        if attention_type:
            possible_params['attention_type'] = attention_type
        for kwargs in kwargs_product(possible_params):
            attention = Attention(self.dimensions, **kwargs)
            for param in attention.parameters():
                param.data.uniform_(-.1, .1)
            yield attention, kwargs

    def test_init(self):
        Attention(self.dimensions)

    def test_forward(self):
        for attention, _ in self._attentions():
            output, weights = attention.forward(self.input_, self.context)

            # Check sizes
            self.assertEqual(output.size(), (self.batch_size, self.output_seq_len, self.dimensions))
            self.assertEqual(weights.size(),
                             (self.batch_size, self.output_seq_len, self.input_seq_len))

            # Check types
            self.assertEqual(output.data.type(), 'torch.FloatTensor')
            self.assertEqual(weights.data.type(), 'torch.FloatTensor')
