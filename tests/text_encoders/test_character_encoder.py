import pickle

import pytest

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import UNKNOWN_TOKEN
from torchnlp.text_encoders.reserved_tokens import RESERVED_ITOS


@pytest.fixture
def sample():
    return ['The quick brown fox jumps over the lazy dog']


@pytest.fixture
def encoder(sample):
    return CharacterEncoder(sample)


def test_character_encoder(encoder, sample):
    input_ = 'english-language pangram'
    output = encoder.encode(input_)
    assert encoder.vocab_size == len(set(list(sample[0]))) + len(RESERVED_ITOS)
    assert len(output) == len(input_)
    assert encoder.decode(output) == input_.replace('-', UNKNOWN_TOKEN)


def test_character_encoder_min_occurrences(sample):
    encoder = CharacterEncoder(sample, min_occurrences=10)
    input_ = 'English-language pangram'
    output = encoder.encode(input_)
    assert encoder.decode(output) == ''.join([UNKNOWN_TOKEN] * len(input_))


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
