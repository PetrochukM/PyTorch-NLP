from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s):
    return list(s)


class CharacterEncoder(StaticTokenizerEncoder):
    """ Encodes text into a tensor by splitting the text into individual characters.

    Args:
        sample (list): Sample of data used to build encoding dictionary.
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          the encoding dictionary.
        append_eos (bool, optional): If ``True`` append EOS token onto the end to the encoded
          vector.
        reserved_tokens (list of str, optional): List of reserved tokens inserted in the beginning
            of the dictionary.
        eos_index (int, optional): The eos token is used to encode the end of a sequence. This is
          the index that token resides at.
        unknown_index (int, optional): The unknown token is used to encode unseen tokens. This is
          the index that token resides at.
        padding_index (int, optional): The unknown token is used to encode sequence padding. This is
          the index that token resides at.
    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('Encoder does not take keyword argument tokenize.')

        super().__init__(*args, tokenize=_tokenize, **kwargs)

    def decode(self, tensor):
        return ''.join([self.itos[index] for index in tensor])
