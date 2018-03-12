from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class IdentityEncoder(StaticTokenizerEncoder):
    """ Encodes the text without tokenization.

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:

        >>> encoder = IdentityEncoder(['label_a', 'label_b'])
        >>> encoder.encode('label_a')
         5
        [torch.LongTensor of size 1]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'label_a', 'label_b']
        >>>
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
        if 'append_eos' not in kwargs:
            kwargs['append_eos'] = False  # Default to not appending EOS
        super().__init__(*args, **kwargs, tokenize=(lambda s: s if isinstance(s, list) else [s]))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        if len(tokens) == 1:
            return tokens[0]
        else:
            return tokens
