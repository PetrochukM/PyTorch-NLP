import torch

from torchnlp.text_encoders.reserved_tokens import EOS_INDEX
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_INDEX
from torchnlp.text_encoders.reserved_tokens import RESERVED_ITOS
from torchnlp.text_encoders.reserved_tokens import RESERVED_STOI
from torchnlp.text_encoders.subword_text_tokenizer import SubwordTextTokenizer
from torchnlp.text_encoders.text_encoders import TextEncoder


class SubwordEncoder(TextEncoder):
    """ Use Googles Tensor2Tensor SubwordTextTokenizer """

    def __init__(self,
                 sample,
                 append_eos=False,
                 lower=True,
                 target_vocab_size=None,
                 min_occurrences=1,
                 max_occurrences=1e3):
        """ Given a sample, build the dictionary for the word encoder.

        Args:
            sample (list of str)
            append_eos (bool)
            lower (bool)
            target_vocab_size (int): desired vocab_size to approximate
            min_occurrences (int): lower bound for the minimum token count
            max_occurrences (int): upper bound for the minimum token count
        """
        self.lower = lower
        self.append_eos = append_eos

        if self.lower:
            sample = [text.lower().rstrip('\n') for text in sample]

        if target_vocab_size is None:
            self.tokenizer = SubwordTextTokenizer()
            self.tokenizer.build_from_corpus(sample, min_count=min_occurrences)
        else:
            self.tokenizer = SubwordTextTokenizer.build_to_target_size_from_corpus(
                sample,
                target_size=target_vocab_size,
                min_val=min_occurrences,
                max_val=max_occurrences)

        self.stoi = RESERVED_STOI.copy()
        self.itos = RESERVED_ITOS[:]
        for token in self.tokenizer.vocab:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        return self.itos

    def encode(self, text):
        if self.lower:
            text = text.lower()
        text = text.rstrip('\n')
        text = self.tokenizer.encode(text)
        vector = [self.stoi.get(token, UNKNOWN_INDEX) for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.tokenizer.decode(tokens)
