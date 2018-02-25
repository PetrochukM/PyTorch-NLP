import unittest
import random

from lib.datasets import reverse
from lib.datasets import zero_to_zero
from lib.datasets import count


class TestDatasets(unittest.TestCase):

    def test_init(self):
        for dataset in [reverse, zero_to_zero, count]:
            dataset(test=True, train_rows=32, dev_rows=32, test_rows=32)

    def test_continuity(self):
        # Good test for generated datasets
        state = random.getstate()
        for dataset in [reverse, zero_to_zero, count]:
            print(dataset.__name__)
            random.setstate(state)
            test = reverse(test=True, train_rows=32, dev_rows=32, test_rows=32)
            random.setstate(state)
            _, test_1 = reverse(train=True, test=True, train_rows=32, dev_rows=32, test_rows=32)
            self.assertEqual(test.rows, test_1.rows)
            random.setstate(state)
            _, _, test_2 = reverse(
                train=True, dev=True, test=True, train_rows=32, dev_rows=32, test_rows=32)
            self.assertEqual(test_1.rows, test_2.rows)
