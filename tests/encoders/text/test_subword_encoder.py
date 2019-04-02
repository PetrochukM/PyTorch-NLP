import pickle

import pytest

from torchnlp.encoders.text import SubwordEncoder
from torchnlp.encoders.text import DEFAULT_EOS_INDEX


class TestSubwordEncoder:

    @pytest.fixture(scope='module')
    def corpus(self):
        return [
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx',
            "I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg'
        ]

    @pytest.fixture
    def encoder(self, corpus):
        return SubwordEncoder(corpus, target_vocab_size=86, min_occurrences=2, max_occurrences=6)

    def test_build_vocab_target_size(self, encoder):
        # NOTE: ``target_vocab_size`` is approximate; therefore, it may not be exactly the target
        # size
        assert len(encoder.vocab) == 86

    def test_encode(self, encoder):
        input_ = 'This has UPPER CASE letters that are out of alphabet'
        assert encoder.decode(encoder.encode(input_)) == input_

    def test_eos(self, corpus):
        encoder = SubwordEncoder(corpus, append_eos=True)
        input_ = 'This is a sentence'
        assert encoder.encode(input_)[-1] == DEFAULT_EOS_INDEX

    def test_is_pickleable(self, encoder):
        pickle.dumps(encoder)
