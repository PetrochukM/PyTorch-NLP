import unittest

from lib.metrics import get_token_accuracy
from tests.lib.utils import get_batch


class TestGetTokenAccuracy(unittest.TestCase):

    def setUp(self):
        _, targets, outputs = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        self.targets = targets
        self.outputs = outputs

    def test_ignore_index_none(self):
        accuracy, n_correct, n_total = get_token_accuracy(self.targets, self.outputs, print_=True)
        self.assertAlmostEqual(accuracy, 0.75)
        self.assertAlmostEqual(n_correct, 3)
        self.assertAlmostEqual(n_total, 4)

    def test_ignore_index(self):
        accuracy, n_correct, n_total = get_token_accuracy(
            self.targets, self.outputs, ignore_index=1)
        self.assertAlmostEqual(accuracy, 1)
        self.assertAlmostEqual(n_correct, 3)
        self.assertAlmostEqual(n_total, 3)
