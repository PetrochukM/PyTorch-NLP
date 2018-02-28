from torchnlp.text_encoders import SpacyEncoder


def test_spacy_encoder():
    input_ = 'This is a sentence'
    encoder = SpacyEncoder([input_], lower=False)
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_
