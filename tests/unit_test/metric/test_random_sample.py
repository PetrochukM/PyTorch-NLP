import unittest

from lib.metrics import print_random_sample
from tests.lib.utils import get_batch
from lib.text_encoders import WordEncoder


class TestPrintRandomSample(unittest.TestCase):

    def setUp(self):
        self.output_text_encoder = WordEncoder(['a b c d e'], append_eos=False)
        self.input_text_encoder = WordEncoder(['a b c d e'], append_eos=False)
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
        sources, targets, outputs = get_batch(
            predictions=predictions,
            targets=targets,
            vocab_size=self.output_text_encoder.vocab_size)
        self.sources = sources
        self.targets = targets
        self.outputs = outputs

    def test_ignore_index_none(self):
        print_random_sample(
            self.sources,
            self.targets,
            self.outputs,
            self.input_text_encoder,
            self.output_text_encoder,
            n_samples=1)

    def test_ignore_index(self):
        print_random_sample(
            self.sources,
            self.targets,
            self.outputs,
            self.input_text_encoder,
            self.output_text_encoder,
            n_samples=1,
            ignore_index=self.output_text_encoder.stoi['e'])

    def test_n_samples_big(self):
        print_random_sample(
            self.sources,
            self.targets,
            self.outputs,
            self.input_text_encoder,
            self.output_text_encoder,
            n_samples=40)
