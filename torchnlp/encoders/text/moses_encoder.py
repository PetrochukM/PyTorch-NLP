from functools import partial

from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


class MosesEncoder(StaticTokenizerEncoder):
    """ Encodes the text using the Moses tokenizer.

    **Tokenizer Reference:**
    http://www.nltk.org/_modules/nltk/tokenize/moses.html

    Args:
        **args: Arguments passed onto ``StaticTokenizerEncoder.__init__``.
        **kwargs: Keyword arguments passed onto ``StaticTokenizerEncoder.__init__``.

    Example:

        >>> encoder = MosesEncoder(["This ain't funny.", "Don't?"])
        >>> encoder.encode("This ain't funny.")
        tensor([5, 6, 7, 8, 9])
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', 'ain', '&apos;t', 'funny', '.', \
'Don', '?']
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('``MosesEncoder`` does not take keyword argument ``tokenize``.')

        if 'detokenize' in kwargs:
            raise TypeError('``MosesEncoder`` does not take keyword argument ``detokenize``.')

        try:
            from sacremoses import MosesTokenizer
            from sacremoses import MosesDetokenizer
        except ImportError:
            print("Please install SacreMoses. "
                  "See the docs at https://github.com/alvations/sacremoses for more information.")
            raise

        super().__init__(
            *args,
            tokenize=MosesTokenizer().tokenize,
            detokenize=partial(MosesDetokenizer().detokenize, return_str=True),
            **kwargs)
