import pickle

import pytest

from torchnlp.encoders.text import WhitespaceEncoder


@pytest.fixture
def input_():
    return 'This is a sentence'


@pytest.fixture
def encoder(input_):
    return WhitespaceEncoder([input_])


def test_whitespace_encoder(encoder, input_):
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
