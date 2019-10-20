from torchnlp.encoders.text.delimiter_encoder import DelimiterEncoder


class WhitespaceEncoder(DelimiterEncoder):
    """ Encodes a text by splitting on whitespace.

    Args:
        **args: Arguments passed onto ``DelimiterEncoder.__init__``.
        **kwargs: Keyword arguments passed onto ``DelimiterEncoder.__init__``.

    Example:

        >>> encoder = WhitespaceEncoder(["This ain't funny.", "Don't?"])
        >>> encoder.encode("This ain't funny.")
        tensor([6, 7, 8])
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', "Don't?", 'This', "ain't", 'funny.']
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."

    """

    def __init__(self, *args, **kwargs):
        super().__init__(' ', *args, **kwargs)
