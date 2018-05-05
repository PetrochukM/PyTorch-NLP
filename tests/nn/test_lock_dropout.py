import random
import unittest

import torch

from torchnlp.nn import LockedDropout


class TestLockedDropout(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.input_ = torch.FloatTensor(
            random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        self.probability = random.random()

    def test_init(self):
        LockedDropout(self.probability)

    def test_forward(self):
        dropout = LockedDropout(self.probability)
        output = dropout.forward(self.input_)

        # Check sizes
        self.assertEqual(output.size(), self.input_.size())

        # Check types
        self.assertEqual(output.type(), 'torch.FloatTensor')
