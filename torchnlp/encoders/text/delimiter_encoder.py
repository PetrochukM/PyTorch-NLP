from functools import partial

from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s, delimiter):
    return s.split(delimiter)


def _detokenize(s, delimiter):
    return delimiter.join(s)


class DelimiterEncoder(StaticTokenizerEncoder):
    """ Encodes text into a tensor by splitting the text using a delimiter.

    Args:
        delimiter (string): Delimiter used with ``string.split``.
        **args: Arguments passed onto ``StaticTokenizerEncoder.__init__``.
        **kwargs: Keyword arguments passed onto ``StaticTokenizerEncoder.__init__``.

    Example:

        >>> encoder = DelimiterEncoder('|', ['token_a|token_b', 'token_c'])
        >>> encoder.encode('token_a|token_c')
        tensor([5, 7])
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'token_a', 'token_b', 'token_c']

    """

    def __init__(self, delimiter, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('``DelimiterEncoder`` does not take keyword argument ``tokenize``.')

        if 'detokenize' in kwargs:
            raise TypeError('``DelimiterEncoder`` does not take keyword argument ``detokenize``.')

        self.delimiter = delimiter

        super().__init__(
            *args,
            tokenize=partial(_tokenize, delimiter=self.delimiter),
            detokenize=partial(_detokenize, delimiter=self.delimiter),
            **kwargs)
