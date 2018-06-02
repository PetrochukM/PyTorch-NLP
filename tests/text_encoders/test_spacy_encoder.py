import pickle

import pytest

from torchnlp.text_encoders import SpacyEncoder


@pytest.fixture
def encoder():
    input_ = 'This is a sentence'
    return SpacyEncoder([input_])


def test_spacy_encoder(encoder):
    input_ = 'This is a sentence'
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_


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

    assert error_message.startswith("No tokenizer available for language " +
                                    "'python'.")


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
