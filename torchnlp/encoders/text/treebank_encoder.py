from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


class TreebankEncoder(StaticTokenizerEncoder):
    """ Encodes text using the Treebank tokenizer.

    **Tokenizer Reference:**
    http://www.nltk.org/_modules/nltk/tokenize/treebank.html

    Args:
        **args: Arguments passed onto ``StaticTokenizerEncoder.__init__``.
        **kwargs: Keyword arguments passed onto ``StaticTokenizerEncoder.__init__``.

    Example:

        >>> encoder = TreebankEncoder(["This ain't funny.", "Don't?"])
        >>> encoder.encode("This ain't funny.")
        tensor([5, 6, 7, 8, 9])
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', 'ai', "n't", 'funny', '.', 'Do', '?']
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('``TreebankEncoder`` does not take keyword argument ``tokenize``.')

        if 'detokenize' in kwargs:
            raise TypeError('``TreebankEncoder`` does not take keyword argument ``detokenize``.')

        try:
            import nltk

            # Required for moses
            nltk.download('perluniprops')
            nltk.download('nonbreaking_prefixes')

            from nltk.tokenize.treebank import TreebankWordTokenizer
            from nltk.tokenize.treebank import TreebankWordDetokenizer
        except ImportError:
            print("Please install NLTK. " "See the docs at http://nltk.org for more information.")
            raise

        super().__init__(
            *args,
            tokenize=TreebankWordTokenizer().tokenize,
            detokenize=TreebankWordDetokenizer().detokenize,
            **kwargs)
