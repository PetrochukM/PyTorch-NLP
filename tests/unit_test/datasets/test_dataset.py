import unittest

from lib.datasets.dataset import Dataset


class SeqInputOutputDatasetTest(unittest.TestCase):

    def test_init(self):
        data = Dataset([{}])

    def test_raises_unequal_columns(self):
        self.assertRaises(AssertionError, lambda: Dataset([{'a': 'a'}, {'b': 'b'}]))

    def test_len(self):
        data = Dataset([{'a': 'a'}, {'a': 'a'}])
        self.assertEqual(len(data), 2)

    def test_get_item(self):
        data = Dataset([{'a': 'a'}, {'a': 'b'}])
        self.assertEqual(data[0], {'a': 'a'})

    def test_contains(self):
        data = Dataset([{'a': 'a'}, {'a': 'b'}])
        self.assertTrue('a' in data)
