from collections import Counter

from torchnlp.encoders.encoder import Encoder

import torch

DEFAULT_UNKNOWN_TOKEN = '<unk>'
DEFAULT_RESERVED = [DEFAULT_UNKNOWN_TOKEN]


class LabelEncoder(Encoder):
    """ Encodes an label via a dictionary.

    Args:
        sample (list of strings): Sample of data used to build encoding dictionary.
        min_occurrences (int, optional): Minimum number of occurrences for a label to be added to
          the encoding dictionary.
        reserved_labels (list, optional): List of reserved labels inserted in the beginning of the
          dictionary.
        unknown_index (int, optional): The unknown label is used to encode unseen labels. This is
          the index that label resides at.
        **kwargs: Keyword arguments passed onto ``Encoder``.

    Example:

        >>> samples = ['label_a', 'label_b']
        >>> encoder = LabelEncoder(samples, reserved_labels=['unknown'], unknown_index=0)
        >>> encoder.encode('label_a')
        tensor(1)
        >>> encoder.decode(encoder.encode('label_a'))
        'label_a'
        >>> encoder.encode('label_c')
        tensor(0)
        >>> encoder.decode(encoder.encode('label_c'))
        'unknown'
        >>> encoder.vocab
        ['unknown', 'label_a', 'label_b']
    """

    def __init__(self,
                 sample,
                 min_occurrences=1,
                 reserved_labels=DEFAULT_RESERVED,
                 unknown_index=DEFAULT_RESERVED.index(DEFAULT_UNKNOWN_TOKEN),
                 **kwargs):
        super().__init__(**kwargs)

        if unknown_index and unknown_index >= len(reserved_labels):
            raise ValueError('The `unknown_index` if provided must be also `reserved`.')

        self.unknown_index = unknown_index
        self.tokens = Counter(sample)
        self.itos = reserved_labels.copy()
        self.stoi = {token: index for index, token in enumerate(reserved_labels)}
        for token, count in self.tokens.items():
            if count >= min_occurrences:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        """
        Returns:
            list: List of labels in the dictionary.
        """
        return self.itos

    @property
    def vocab_size(self):
        """
        Returns:
            int: Number of labels in the dictionary.
        """
        return len(self.vocab)

    def encode(self, label):
        """ Encodes a ``label``.

        Args:
            label (object): Label to encode.

        Returns:
            torch.Tensor: Encoding of the label.
        """
        label = super().encode(label)

        return torch.tensor(self.stoi.get(label, self.unknown_index))

    def batch_encode(self, iterator, *args, dim=0, **kwargs):
        """
        Args:
            iterator (iterator): Batch of labels to encode.
            *args: Arguments passed to ``Encoder.batch_encode``.
            dim (int, optional): Dimension along which to concatenate tensors.
            **kwargs: Keyword arguments passed to ``Encoder.batch_encode``.

        Returns:
            torch.Tensor: Tensor of encoded labels.
        """
        return torch.stack(super().batch_encode(iterator, *args, **kwargs), dim=dim)

    def decode(self, encoded):
        """ Decodes ``encoded`` label.

        Args:
            encoded (torch.Tensor): Encoded label.

        Returns:
            object: Label decoded from ``encoded``.
        """
        encoded = super().decode(encoded)

        if encoded.numel() > 1:
            raise ValueError(
                '``decode`` decodes one label at a time, use ``batch_decode`` instead.')

        return self.itos[encoded.squeeze().item()]

    def batch_decode(self, tensor, *args, dim=0, **kwargs):
        """
        Args:
            tensor (torch.Tensor): Batch of tensors.
            *args: Arguments passed to ``Encoder.batch_decode``.
            dim (int, optional): Dimension along which to split tensors.
            **kwargs: Keyword arguments passed to ``Encoder.batch_decode``.

        Returns:
            list: Batch of decoded labels.
        """
        return super().batch_decode([t.squeeze(0) for t in tensor.split(1, dim=dim)])
