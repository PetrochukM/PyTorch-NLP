import pickle

import pytest

from torchnlp.encoders.text import TreebankEncoder


@pytest.fixture
def input_():
    return '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''


@pytest.fixture
def encoder(input_):
    return TreebankEncoder([input_])


def test_treebank_encoder(encoder, input_):
    # TEST adapted from example in http://www.nltk.org/_modules/nltk/tokenize/treebank.html
    expected_tokens = [
        'Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',
        'of', 'them.', 'Thanks', '.'
    ]
    expected_decode = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
    tokens = encoder.encode(input_)
    assert [encoder.itos[i] for i in tokens] == expected_tokens
    assert encoder.decode(tokens) == expected_decode


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
