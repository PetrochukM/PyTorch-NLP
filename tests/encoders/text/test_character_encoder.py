import pickle

import pytest

from torchnlp.encoders.text import CharacterEncoder
from torchnlp.encoders.text import DEFAULT_RESERVED_TOKENS
from torchnlp.encoders.text import DEFAULT_UNKNOWN_TOKEN


@pytest.fixture
def sample():
    return ['The quick brown fox jumps over the lazy dog']


@pytest.fixture
def encoder(sample):
    return CharacterEncoder(sample)


def test_character_encoder(encoder, sample):
    input_ = 'english-language pangram'
    output = encoder.encode(input_)
    assert encoder.vocab_size == len(set(list(sample[0]))) + len(DEFAULT_RESERVED_TOKENS)
    assert len(output) == len(input_)
    assert encoder.decode(output) == input_.replace('-', DEFAULT_UNKNOWN_TOKEN)


def test_character_encoder_batch(encoder, sample):
    input_ = 'english-language pangram'
    longer_input_ = 'english-language pangram pangram'
    encoded, lengths = encoder.batch_encode([input_, longer_input_])
    assert encoded.shape[0] == 2
    assert len(lengths) == 2
    decoded = encoder.batch_decode(encoded, lengths=lengths)
    assert decoded[0] == input_.replace('-', DEFAULT_UNKNOWN_TOKEN)
    assert decoded[1] == longer_input_.replace('-', DEFAULT_UNKNOWN_TOKEN)


def test_character_encoder_min_occurrences(sample):
    encoder = CharacterEncoder(sample, min_occurrences=10)
    input_ = 'English-language pangram'
    output = encoder.encode(input_)
    assert encoder.decode(output) == ''.join([DEFAULT_UNKNOWN_TOKEN] * len(input_))


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
