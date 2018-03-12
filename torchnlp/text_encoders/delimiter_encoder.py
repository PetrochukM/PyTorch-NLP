from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder
from torchnlp.text_encoders.reserved_tokens import RESERVED_STOI
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_TOKEN


class DelimiterEncoder(StaticTokenizerEncoder):
    """ Encodes text into a tensor by splitting the text using a delimiter.

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:

        >>> encoder = DelimiterEncoder('|', ['token_a|token_b', 'token_c'])
        >>> encoder.encode('token_a|token_c')
         5
         7
        [torch.LongTensor of size 2]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'token_a', 'token_b', 'token_c']

    """

    def __init__(self, delimiter, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('CharacterEncoder defines a tokenize callable per character')
        self.delimiter = delimiter
        super().__init__(*args, **kwargs, tokenize=(lambda s: s.split(delimiter)))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]

        # NOTE: Join reserved tokens like `PAD_TOKEN` and `EOS_TOKEN` with '' instead of delimiter
        # for aesthetic reasons at the end of the text phrase
        reserved = []
        while len(tokens) > 0 and tokens[-1] in RESERVED_STOI and tokens[-1] != UNKNOWN_TOKEN:
            reserved.insert(0, tokens.pop())

        return self.delimiter.join(tokens) + ''.join(reserved)
