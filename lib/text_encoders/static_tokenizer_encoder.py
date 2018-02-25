import torch

from lib.text_encoders.reserved_tokens import EOS_INDEX
from lib.text_encoders.reserved_tokens import UNKNOWN_INDEX
from lib.text_encoders.reserved_tokens import RESERVED_ITOS
from lib.text_encoders.reserved_tokens import RESERVED_STOI
from lib.text_encoders.text_encoders import TextEncoder

# TODO: Think about should the encoder during decoding include <eos> or not?


class StaticTokenizerEncoder(TextEncoder):
    """ Encoder where the tokenizer is not learned and a static function. """

    def __init__(self, sample, append_eos=False, lower=True, tokenize=(lambda s: s.split())):
        """ Given a sample, build the dictionary for the word encoder """
        self.lower = lower
        self.tokenize = tokenize
        self.append_eos = append_eos
        self.tokens = set()

        for text in sample:
            self.tokens.update(self._preprocess(text))

        self.stoi = RESERVED_STOI.copy()
        self.itos = RESERVED_ITOS[:]
        for token in self.tokens:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
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
        text = self._preprocess(text)
        vector = [self.stoi.get(token, UNKNOWN_INDEX) for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
