from collections import Counter

import torch

from torchnlp.text_encoders.reserved_tokens import EOS_INDEX
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_INDEX
from torchnlp.text_encoders.reserved_tokens import RESERVED_ITOS
from torchnlp.text_encoders.reserved_tokens import RESERVED_STOI
from torchnlp.text_encoders.text_encoder import TextEncoder


class StaticTokenizerEncoder(TextEncoder):
    """ Encodes the text using a lambda tokenizer.

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.
        tokenize (callable): callable to tokenize a string

    Example:

        >>> encoder = StaticTokenizerEncoder(["This ain't funny.", "Don't?"],
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

    def __init__(self, sample, min_occurrences=1, append_eos=False, tokenize=(lambda s: s.split())):
        if not isinstance(sample, list):
            raise TypeError('Sample must be a list of strings.')

        self.append_eos = append_eos
        self.tokens = Counter()
        self.tokenize = tokenize if tokenize else lambda x: x

        for text in sample:
            self.tokens.update(self.tokenize(text))

        self.stoi = RESERVED_STOI.copy()
        self.itos = RESERVED_ITOS[:]
        for token, count in self.tokens.items():
            if count >= min_occurrences:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        return self.itos

    def encode(self, text):
        text = self.tokenize(text)
        vector = [self.stoi.get(token, UNKNOWN_INDEX) for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
