from collections import Counter

import torch

from torchnlp.text_encoders.reserved_tokens import EOS_INDEX
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_INDEX
from torchnlp.text_encoders.reserved_tokens import RESERVED_ITOS
from torchnlp.text_encoders.reserved_tokens import RESERVED_STOI
from torchnlp.text_encoders.text_encoders import TextEncoder


class StaticTokenizerEncoder(TextEncoder):
    """ Encoder where the tokenizer is a static callable. """

    def __init__(self,
                 sample,
                 min_occurrences=1,
                 append_eos=False,
                 lower=True,
                 tokenize=(lambda s: s.split())):
        """
        Given a sample, build the dictionary for the word encoder.

        Args:
            sample (list of strings): sample of data to build dictionary on
            min_occurrences (int): minimum number of occurrences for a token to be added to
              dictionary
            append_eos (bool): if to append EOS token onto the end to the encoded vector
            lower (bool): if to lower the text
            tokenize (callable): callable to tokenize a string
        """
        if not isinstance(sample, list):
            raise TypeError('Sample needs to be a list of strings.')

        self.lower = lower
        self.tokenize = tokenize
        self.append_eos = append_eos
        self.tokens = Counter()

        for text in sample:
            self.tokens.update(self._preprocess(text))

        self.stoi = RESERVED_STOI.copy()
        self.itos = RESERVED_ITOS[:]
        for token, count in self.tokens.items():
            if count >= min_occurrences:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        """ Return a list of tokens """
        return self.itos

    def _preprocess(self, text):
        """ Preprocess text before encoding as a tensor. """
        if self.lower:
            text = text.lower()
        text = text.rstrip('\n')
        if self.tokenize:
            text = self.tokenize(text)
        return text

    def encode(self, text):
        """ Encode text into a vector """
        text = self._preprocess(text)
        vector = [self.stoi.get(token, UNKNOWN_INDEX) for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        """ Decode vector into text, not guaranteed to be reversable """
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
