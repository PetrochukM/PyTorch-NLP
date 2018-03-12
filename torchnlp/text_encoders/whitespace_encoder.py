from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class WhitespaceEncoder(StaticTokenizerEncoder):
    """ Encodes the text by splitting on whitespace.

    Tokenization Algorithm Reference:
    https://docs.python.org/3/library/stdtypes.html#str.split

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:

        >>> encoder = WhitespaceEncoder(["This ain't funny.", "Don't?"],
                                             tokenize=lambda s: s.split())
        >>> encoder.encode("This ain't funny.")
         5
         6
         7
        [torch.LongTensor of size 3]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', "ain't", 'funny.', "Don't?"]
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('WhiteSpaceEncoder defines a tokenize callable per character')
        super().__init__(*args, **kwargs, tokenize=(lambda s: s.split()))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
