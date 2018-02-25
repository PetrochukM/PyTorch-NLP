import unittest

from lib.metrics import print_bucket_accuracy
from tests.lib.utils import get_batch
from lib.text_encoders import WordEncoder


class TestPrintBucketAccuracy(unittest.TestCase):

    def setUp(self):
        self.output_text_encoder = WordEncoder(['a b c d e'], append_eos=False)
        predictions = [
            self.output_text_encoder.encode('a b c d d').tolist(),
            self.output_text_encoder.encode('a a a a a').tolist(),
            self.output_text_encoder.encode('b b b b b').tolist(),
        ]
        targets = [
            self.output_text_encoder.encode('a b c d e').tolist(),
            self.output_text_encoder.encode('a a a a a').tolist(),
            self.output_text_encoder.encode('b b b b b').tolist(),
        ]
        _, targets, outputs = get_batch(
            predictions=predictions,
            targets=targets,
            vocab_size=self.output_text_encoder.vocab_size)
        self.buckets = [1, 2, 2]
        self.targets = targets
        self.outputs = outputs

    def test_ignore_index_none(self):
        print_bucket_accuracy(self.buckets, self.targets, self.outputs)

    def test_ignore_index(self):
        print_bucket_accuracy(
            self.buckets,
            self.targets,
            self.outputs,
            ignore_index=self.output_text_encoder.stoi['e'])
