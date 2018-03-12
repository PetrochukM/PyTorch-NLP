from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class CharacterEncoder(StaticTokenizerEncoder):
    """ Encodes text into a tensor by splitting the text into individual characters.

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.
    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('CharacterEncoder defines a tokenize callable per character')
        super().__init__(*args, **kwargs, tokenize=(lambda s: list(s)))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ''.join(tokens)
