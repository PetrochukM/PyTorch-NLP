from collections import namedtuple

import torch

from torchnlp.encoders import Encoder
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX


def pad_tensor(tensor, length, padding_index=DEFAULT_PADDING_INDEX):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.

    Args:
        tensor (torch.Tensor [n, ...]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.

    Returns
        (torch.Tensor [length, ...]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    assert n_padding >= 0
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)


BatchedSequences = namedtuple('BatchedSequences', ['tensor', 'lengths'])


def stack_and_pad_tensors(batch, padding_index=DEFAULT_PADDING_INDEX, dim=0):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.
        dim (int, optional): Dimension on to which to concatenate the batch of tensors.

    Returns
        BatchedSequences(torch.Tensor, torch.Tensor): Padded tensors and original lengths of
            tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    lengths = torch.tensor(lengths)
    padded = torch.stack(padded, dim=dim).contiguous()
    for _ in range(dim):
        lengths = lengths.unsqueeze(0)

    return BatchedSequences(padded, lengths)


class TextEncoder(Encoder):

    def decode(self, encoded):
        """ Decodes an object.

        Args:
            object_ (object): Encoded object.

        Returns:
            object: Object decoded.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            decoded_encoded = self.encode(self.decode(encoded))
            self.enforce_reversible = True
            if not torch.equal(decoded_encoded, encoded):
                raise ValueError('Decoding is not reversible for "%s"' % encoded)

        return encoded

    def batch_encode(self, iterator, *args, dim=0, **kwargs):
        """
        Args:
            iterator (iterator): Batch of text to encode.
            *args: Arguments passed onto ``Encoder.__init__``.
            dim (int, optional): Dimension along which to concatenate tensors.
            **kwargs: Keyword arguments passed onto ``Encoder.__init__``.

        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        return stack_and_pad_tensors(
            super().batch_encode(iterator), padding_index=self.padding_index, dim=dim)

    def batch_decode(self, tensor, lengths, dim=0, *args, **kwargs):
        """
        Args:
            batch (list of :class:`torch.Tensor`): Batch of encoded sequences.
            lengths (torch.Tensor): Original lengths of sequences.
            dim (int, optional): Dimension along which to split tensors.
            *args: Arguments passed to ``decode``.
            **kwargs: Key word arguments passed to ``decode``.

        Returns:
            list: Batch of decoded sequences.
        """
        return super().batch_decode(
            [t.squeeze(0)[:l] for t, l in zip(tensor.split(1, dim=dim), lengths)])
