from torchnlp.text_encoders import WhitespaceEncoder


def test_spacy_encoder():
    input_ = 'This is a sentence'
    encoder = WhitespaceEncoder([input_])
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_
