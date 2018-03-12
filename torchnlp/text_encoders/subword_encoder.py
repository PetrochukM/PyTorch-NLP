import torch

from torchnlp.text_encoders.reserved_tokens import EOS_INDEX
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_INDEX
from torchnlp.text_encoders.reserved_tokens import RESERVED_ITOS
from torchnlp.text_encoders.reserved_tokens import RESERVED_STOI
from torchnlp.text_encoders.subword_text_tokenizer import SubwordTextTokenizer
from torchnlp.text_encoders.text_encoder import TextEncoder


class SubwordEncoder(TextEncoder):
    """ Invertibly encoding text using a limited vocabulary.

    Applies Googles Tensor2Tensor SubwordTextTokenizer that invertibly encodes a native string as a
    sequence of subtokens from a limited vocabulary. In order to build the vocabulary, it uses
    recursive binary search to find a minimum token count `x`
    (s.t. `min_occurrences` <= `x` <= `max_occurrences`) that most closely matches the
    `target_size`.

    Tokenization Algorithm Reference:
    https://github.com/tensorflow/tensor2tensor/blob/8bdecbe434d93cb1e79c0489df20fee2d5a37dc2/tensor2tensor/data_generators/text_encoder.py#L389

    Args:
        sample (list of strings): Sample of data to build dictionary on
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.
        target_vocab_size (int, optional): Desired size of vocab.
        min_occurrences (int, optional): Lower bound for the minimum token count.
        max_occurrences (int, optional): Upper bound for the minimum token count.
    """

    def __init__(self,
                 sample,
                 append_eos=False,
                 target_vocab_size=None,
                 min_occurrences=1,
                 max_occurrences=1e3):
        self.append_eos = append_eos

        if target_vocab_size is None:
            self.tokenizer = SubwordTextTokenizer()
            self.tokenizer.build_from_corpus(sample, min_count=min_occurrences)
        else:

            target_vocab_size -= len(RESERVED_ITOS)
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
        text = self.tokenizer.encode(text)
        vector = [self.stoi.get(token, UNKNOWN_INDEX) for token in text]
        if self.append_eos:
            vector.append(EOS_INDEX)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.tokenizer.decode(tokens)
