from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s):
    return s if isinstance(s, list) else [s]


class IdentityEncoder(StaticTokenizerEncoder):
    """ Encodes the text without tokenization.

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:
        >>> encoder = IdentityEncoder(['token_a', 'token_b', 'token_c'])
        >>> encoder.encode(['token_a', 'token_b'])
         5
         6
        [torch.LongTensor of size 2]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'token_a', 'token_b', 'token_c']
    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('IdentityEncoder defines a identity tokenization')
        super().__init__(*args, tokenize=_tokenize, **kwargs)

    def decode(self, tensor):
        if len(tensor.shape) == 0:
            tensor = tensor.unsqueeze(0)

        tokens = [self.itos[index] for index in tensor]
        if len(tokens) == 1:
            return tokens[0]
        else:
            return tokens
