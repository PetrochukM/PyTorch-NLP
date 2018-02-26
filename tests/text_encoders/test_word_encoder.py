from lib.text_encoders import WordEncoder


def test_spacy_encoder():
    input_ = 'This is a sentence'
    encoder = WordEncoder([input_], lower=False)
    tokens = encoder.encode(input_)
    assert encoder.decode(tokens) == input_
