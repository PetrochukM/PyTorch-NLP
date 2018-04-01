import random
import unittest

import torch

from torchnlp.nn import LockedDropout
from tests.nn.utils import tensor


class TestLockedDropout(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.input_ = tensor(
            random.randint(1, 10),
            random.randint(1, 10),
            random.randint(1, 10),
            type_=torch.FloatTensor)
        self.probability = random.random()

    def test_init(self):
        LockedDropout(self.probability)

    def test_forward(self):
        dropout = LockedDropout(self.probability)
        output = dropout.forward(self.input_)

        # Check sizes
        self.assertEqual(output.size(), self.input_.size())

        # Check types
        self.assertEqual(output.data.type(), 'torch.FloatTensor')
