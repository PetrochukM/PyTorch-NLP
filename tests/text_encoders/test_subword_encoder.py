import unittest

from torchnlp.text_encoders import SubwordEncoder
from torchnlp.text_encoders import EOS_INDEX


class SubwordEncoderTest(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx',
            "I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg'
        ]

    def test_build_vocab_target_size(self):
        encoder = SubwordEncoder(
            self.corpus, target_vocab_size=81, min_occurrences=2, max_occurrences=6)
        assert len(encoder.vocab) == 81

    def test_encode(self):
        encoder = SubwordEncoder(
            self.corpus, target_vocab_size=81, min_occurrences=2, max_occurrences=6, lower=False)
        input_ = 'This has UPPER CASE letters that are out of alphabet'
        assert encoder.decode(encoder.encode(input_)) == input_

    def test_encode_lower(self):
        encoder = SubwordEncoder(
            self.corpus, target_vocab_size=81, min_occurrences=2, max_occurrences=6, lower=True)
        input_ = 'The quick brown fox jumps over the lazy dog'
        assert encoder.decode(encoder.encode(input_)) == input_.lower()

    def test_eos(self):
        encoder = SubwordEncoder(self.corpus, append_eos=True)
        input_ = 'This is a sentence'
        assert encoder.encode(input_)[-1] == EOS_INDEX
