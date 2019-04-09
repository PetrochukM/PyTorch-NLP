import torch

from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_RESERVED_TOKENS
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_UNKNOWN_INDEX
from torchnlp.encoders.text.text_encoder import TextEncoder
from torchnlp.encoders.text.subword_text_tokenizer import SubwordTextTokenizer


class SubwordEncoder(TextEncoder):
    """ Invertibly encoding text using a limited vocabulary.

    Applies Googles Tensor2Tensor ``SubwordTextTokenizer`` that invertibly encodes a native string
    as a
    sequence of subtokens from a limited vocabulary. In order to build the vocabulary, it uses
    recursive binary search to find a minimum token count `x`
    (s.t. `min_occurrences` <= `x` <= `max_occurrences`) that most closely matches the
    `target_size`.

    **Tokenizer Reference:**
    https://github.com/tensorflow/tensor2tensor/blob/8bdecbe434d93cb1e79c0489df20fee2d5a37dc2/tensor2tensor/data_generators/text_encoder.py#L389

    Args:
        sample (list): Sample of data used to build encoding dictionary.
        append_eos (bool, optional): If ``True`` append EOS token onto the end to the encoded
            vector.
        target_vocab_size (int, optional): Desired size of vocab.
        min_occurrences (int, optional): Lower bound for the minimum token count.
        max_occurrences (int, optional): Upper bound for the minimum token count.
        reserved_tokens (list of str, optional): List of reserved tokens inserted in the beginning
            of the dictionary.
        eos_index (int, optional): The eos token is used to encode the end of a sequence. This is
          the index that token resides at.
        unknown_index (int, optional): The unknown token is used to encode unseen tokens. This is
          the index that token resides at.
        padding_index (int, optional): The unknown token is used to encode sequence padding. This is
          the index that token resides at.
        **kwargs: Keyword arguments passed onto ``TextEncoder.__init__``.
    """

    def __init__(self,
                 sample,
                 append_eos=False,
                 target_vocab_size=None,
                 min_occurrences=1,
                 max_occurrences=1e3,
                 reserved_tokens=DEFAULT_RESERVED_TOKENS,
                 eos_index=DEFAULT_EOS_INDEX,
                 unknown_index=DEFAULT_UNKNOWN_INDEX,
                 padding_index=DEFAULT_PADDING_INDEX,
                 **kwargs):
        super().__init__(**kwargs)

        self.append_eos = append_eos
        self.eos_index = eos_index
        self.unknown_index = unknown_index
        self.reserved_tokens = reserved_tokens
        self.padding_index = padding_index

        if target_vocab_size is None:
            self.tokenizer = SubwordTextTokenizer()
            self.tokenizer.build_from_corpus(sample, min_count=min_occurrences)
        else:

            target_vocab_size -= len(reserved_tokens)
            self.tokenizer = SubwordTextTokenizer.build_to_target_size_from_corpus(
                sample,
                target_size=target_vocab_size,
                min_val=min_occurrences,
                max_val=max_occurrences)

        self.itos = reserved_tokens.copy()
        self.stoi = {token: index for index, token in enumerate(reserved_tokens)}
        for token in self.tokenizer.vocab:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.itos

    @property
    def vocab_size(self):
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return len(self.vocab)

    def encode(self, sequence):
        """ Encodes a ``sequence``.

        Args:
            sequence (str): String ``sequence`` to encode.

        Returns:
            torch.Tensor: Encoding of the ``sequence``.
        """
        sequence = super().encode(sequence)
        sequence = self.tokenizer.encode(sequence)
        vector = [self.stoi.get(token, self.unknown_index) for token in sequence]
        if self.append_eos:
            vector.append(self.eos_index)
        return torch.tensor(vector)

    def decode(self, encoded):
        """ Decodes a tensor into a sequence.

        Args:
            encoded (torch.Tensor): Encoded sequence.

        Returns:
            str: Sequence decoded from ``encoded``.
        """
        encoded = super().decode(encoded)
        return self.tokenizer.decode([self.itos[index] for index in encoded])
