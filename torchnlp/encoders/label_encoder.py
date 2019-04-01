from collections import Counter

from torchnlp.encoder import Encoder

import torch

# RESERVED TOKENS
# NOTE: vocab size is len(reversed) + len(vocab)
UNKNOWN_INDEX = 0
UNKNOWN_TOKEN = '<unk>'
RESERVED_ITOS = [UNKNOWN_TOKEN]
RESERVED_STOI = {token: index for index, token in enumerate(RESERVED_ITOS)}


class LabelEncoder(Encoder):
    """ Encodes the text without tokenization.

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.

    Example:

        >>> encoder = LabelEncoder(['label_a', 'label_b'])
        >>> encoder.encode('label_a')
         5
        [torch.LongTensor of size 1]
        >>> encoder.vocab
        ['<unk>', 'label_a', 'label_b']
    """

    def __init__(self, sample, min_occurrences=1, reserved_tokens=RESERVED_ITOS):
        if not isinstance(sample, list):
            raise TypeError('Sample must be a list of strings.')

        self.tokens = Counter(sample)
        self.itos = reserved_tokens.copy()
        self.stoi = {token: index for index, token in enumerate(reserved_tokens)}
        for token, count in self.tokens.items():
            if count >= min_occurrences:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        return self.itos

    def encode(self, text, unknown_index=UNKNOWN_INDEX):
        vector = [self.stoi.get(text, unknown_index)]
        return torch.LongTensor(vector)

    def decode(self, tensor):
        if len(tensor.shape) == 0:
            tensor = tensor.unsqueeze(0)

        tokens = [self.itos[index] for index in tensor]
        if len(tokens) == 1:
            return tokens[0]
        else:
            return tokens
