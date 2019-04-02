import pickle

import pytest

from torchnlp.encoders.text import StaticTokenizerEncoder


@pytest.fixture
def input_():
    return 'This is a sentence'


@pytest.fixture
def encoder(input_):
    return StaticTokenizerEncoder([input_])


def test_static_tokenizer_encoder(encoder, input_):
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_


def test_static_tokenizer_encoder_batch(encoder, input_):
    batched_input = [input_, input_]
    encoded, lengths = encoder.batch_encode(batched_input)
    assert encoder.batch_decode(encoded, lengths=lengths) == batched_input


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
