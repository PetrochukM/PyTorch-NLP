from collections import Counter
from collections.abc import Iterable

import torch

from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_RESERVED_TOKENS
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_UNKNOWN_INDEX
from torchnlp.encoders.text.text_encoder import TextEncoder


def _tokenize(s):
    return s.split()


def _detokenize(t):
    return ' '.join(t)


class StaticTokenizerEncoder(TextEncoder):
    """ Encodes a text sequence using a static tokenizer.

    Args:
        sample (collections.abc.Iterable): Sample of data used to build encoding dictionary.
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          the encoding dictionary.
        tokenize (callable): :class:`callable` to tokenize a sequence.
        detokenize (callable): :class:`callable` to detokenize a sequence.
        append_eos (bool, optional): If ``True`` append EOS token onto the end to the encoded
          vector.
        reserved_tokens (list of str, optional): List of reserved tokens inserted in the beginning
            of the dictionary.
        eos_index (int, optional): The eos token is used to encode the end of a sequence. This is
          the index that token resides at.
        unknown_index (int, optional): The unknown token is used to encode unseen tokens. This is
          the index that token resides at.
        padding_index (int, optional): The unknown token is used to encode sequence padding. This is
          the index that token resides at.
        **kwargs: Keyword arguments passed onto ``TextEncoder.__init__``.

    Example:

        >>> sample = ["This ain't funny.", "Don't?"]
        >>> encoder = StaticTokenizerEncoder(sample, tokenize=lambda s: s.split())
        >>> encoder.encode("This ain't funny.")
        tensor([5, 6, 7])
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', "ain't", 'funny.', "Don't?"]
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."

    """

    def __init__(self,
                 sample,
                 min_occurrences=1,
                 append_eos=False,
                 tokenize=_tokenize,
                 detokenize=_detokenize,
                 reserved_tokens=DEFAULT_RESERVED_TOKENS,
                 eos_index=DEFAULT_EOS_INDEX,
                 unknown_index=DEFAULT_UNKNOWN_INDEX,
                 padding_index=DEFAULT_PADDING_INDEX,
                 **kwargs):
        super().__init__(**kwargs)

        if not isinstance(sample, Iterable):
            raise TypeError('Sample must be a `collections.abc.Iterable`.')

        self.eos_index = eos_index
        self.unknown_index = unknown_index
        self.padding_index = padding_index
        self.reserved_tokens = reserved_tokens
        self.tokenize = tokenize
        self.detokenize = detokenize
        self.append_eos = append_eos
        self.tokens = Counter()

        for sequence in sample:
            self.tokens.update(self.tokenize(sequence))

        self.index_to_token = reserved_tokens.copy()
        self.token_to_index = {token: index for index, token in enumerate(reserved_tokens)}
        for token, count in self.tokens.items():
            if count >= min_occurrences:
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
        sequence = self.tokenize(sequence)
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
        tokens = [self.index_to_token[index] for index in encoded]
        return self.detokenize(tokens)
