from torchnlp.text_encoders import TreebankEncoder


def test_treebank_encoder():
    # TEST adapted from example in http://www.nltk.org/_modules/nltk/tokenize/treebank.html
    input_ = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
    encoder = TreebankEncoder([input_])
    expected_tokens = [
        'Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two',
        'of', 'them.', 'Thanks', '.'
    ]
    expected_decode = "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
    tokens = encoder.encode(input_)
    assert [encoder.itos[i] for i in tokens] == expected_tokens
    assert encoder.decode(tokens) == expected_decode
