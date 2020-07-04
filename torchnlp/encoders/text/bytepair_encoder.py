import torch

from torchnlp.encoders.text import TextEncoder, DEFAULT_RESERVED_TOKENS, DEFAULT_EOS_INDEX, \
    DEFAULT_UNKNOWN_INDEX, DEFAULT_PADDING_INDEX
from torchnlp.encoders.text.bpe_text_tokenizer import BPETextTokenizer


class BPEEncoder(TextEncoder):
    """ Encodes the text using byte pair encoding.

    **Tokenizer Reference:**
    https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/tools/tokenizer.py

    Args:
        item_list (list): list of data used to build encoding dictionary and BPE tokenizer.
            If they come from files, ``from_filenames`` must be true, otherwise false.
        append_eos (bool, optional): If ``True`` append EOS token onto the end to the encoded
            vector.
        min_occurrences (int, optional): Lower bound for the minimum token count.
        num_symbols (int, optional): desired size of encoding dictionary
        from_filenames (bool, optional): whether item_list refers to file names or not
        reserved_tokens (list of str, optional): List of reserved tokens inserted in the beginning
            of the dictionary.
        eos_index (int, optional): The eos token is used to encode the end of a sequence. This is
          the index that token resides at.
        unknown_index (int, optional): The unknown token is used to encode unseen tokens. This is
          the index that token resides at.
        padding_index (int, optional): The padding token is used to encode sequence padding. This is
          the index that token resides at.
        **kwargs: Keyword arguments passed onto ``TextEncoder.__init__``.
    """

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
