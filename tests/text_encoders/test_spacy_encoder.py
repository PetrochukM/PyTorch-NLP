from torchnlp.text_encoders import SpacyEncoder


def test_spacy_encoder():
    input_ = 'This is a sentence'
    encoder = SpacyEncoder([input_])
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_


def test_spacy_encoder_not_installed_language():
    error_message = ''
    try:
        SpacyEncoder([], language='fr')
    except Exception as e:
        error_message = str(e)

    assert error_message.startswith('Language 'fr' not found.')
