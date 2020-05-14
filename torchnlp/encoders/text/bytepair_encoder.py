import torch

from torchnlp.encoders.text import TextEncoder, DEFAULT_RESERVED_TOKENS, DEFAULT_EOS_INDEX, \
    DEFAULT_UNKNOWN_INDEX, DEFAULT_PADDING_INDEX
from torchnlp.encoders.text.bpe_text_tokenizer import BPETextTokenizer


class BPEEncoder(TextEncoder):
    def __init__(self,
                 item_list,
                 append_eos=False,
                 min_occurrences=2,
                 num_symbols=10000,
                 from_filenames=True,
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

        self.tokenizer = BPETextTokenizer('./bpe')
        self.tokenizer.build_from_corpus(item_list, min_count=min_occurrences,
                                         num_symbols=num_symbols, from_filenames=from_filenames)

        self.index_to_token = reserved_tokens.copy()
        self.token_to_index = {token: index for index, token in enumerate(reserved_tokens)}
        for token in self.tokenizer.vocab:
            self.index_to_token.append(token)
            self.token_to_index[token] = len(self.index_to_token) - 1

    @property
    def vocab(self):
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.index_to_token

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
        vector = [self.token_to_index.get(token, self.unknown_index) for token in sequence]
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
        return self.tokenizer.decode([self.index_to_token[index] for index in encoded])
