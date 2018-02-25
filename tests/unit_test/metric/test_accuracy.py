import unittest

from lib.metrics import get_accuracy
from tests.lib.utils import get_batch


class TestGetAccuracy(unittest.TestCase):

    def setUp(self):
        _, targets, outputs = get_batch(predictions=[[2, 0], [2, 0]], targets=[[2, 0], [2, 1]])
        self.targets = targets
        self.outputs = outputs

    def test_ignore_index_none(self):
        accuracy, n_correct, n_total = get_accuracy(self.targets, self.outputs, print_=True)
        self.assertAlmostEqual(accuracy, 0.5)
        self.assertAlmostEqual(n_correct, 1)
        self.assertAlmostEqual(n_total, 2)

    def test_ignore_index(self):
        accuracy, n_correct, n_total = get_accuracy(self.targets, self.outputs, ignore_index=1)
        self.assertAlmostEqual(accuracy, 1)
        self.assertAlmostEqual(n_correct, 2)
        self.assertAlmostEqual(n_total, 2)
