from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder
from torchnlp.text_encoders.reserved_tokens import RESERVED_STOI
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_TOKEN


class DelimiterEncoder(StaticTokenizerEncoder):
    """ Encode by splitting up by a delimiter """

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
