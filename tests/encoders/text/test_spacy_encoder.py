import pickle

import pytest

from torchnlp.encoders.text import SpacyEncoder


@pytest.fixture
def input_():
    return ('This is a sentence')


@pytest.fixture
def encoder(input_):
    return SpacyEncoder([input_])


def test_spacy_encoder(encoder, input_):
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_


def test_spacy_encoder_issue_44():
    # https://github.com/PetrochukM/PyTorch-NLP/issues/44
    encoder = SpacyEncoder(["This ain't funny."])
    assert 'ai' in encoder.vocab
    assert 'n\'t' in encoder.vocab


def test_spacy_encoder_batch(encoder, input_):
    tokens = encoder.batch_encode([input_, input_])
    assert encoder.decode(tokens[0]) == input_
    assert encoder.decode(tokens[1]) == input_


def test_spacy_encoder_not_installed_language():
    error_message = ''
    try:
        SpacyEncoder([], language='fr')
    except Exception as e:
        error_message = str(e)

    assert error_message.startswith("Language 'fr' not found.")


def test_spacy_encoder_unsupported_language():
    error_message = ''
    try:
        SpacyEncoder([], language='python')
    except Exception as e:
        error_message = str(e)

    assert error_message.startswith("No tokenizer available for language " + "'python'.")


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
