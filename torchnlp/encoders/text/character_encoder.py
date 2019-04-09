from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s):
    return list(s)


def _detokenize(s):
    return ''.join(s)


class CharacterEncoder(StaticTokenizerEncoder):
    """ Encodes text into a tensor by splitting the text into individual characters.

    Args:
        **args: Arguments passed onto ``StaticTokenizerEncoder.__init__``.
        **kwargs: Keyword arguments passed onto ``StaticTokenizerEncoder.__init__``.
    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('``CharacterEncoder`` does not take keyword argument ``tokenize``.')

        if 'detokenize' in kwargs:
            raise TypeError('``CharacterEncoder`` does not take keyword argument ``detokenize``.')

        super().__init__(*args, tokenize=_tokenize, detokenize=_detokenize, **kwargs)
