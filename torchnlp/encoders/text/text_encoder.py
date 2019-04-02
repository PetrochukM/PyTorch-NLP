import torch

from torchnlp.encoders import Encoder
from torchnlp.encoders.sequence.default_reserved_tokens import DEFAULT_PADDING_INDEX


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


def pad_batch(batch, padding_index=DEFAULT_PADDING_INDEX, dim=0):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.
        dim (int, optional): Dimension on to which to concatenate the batch of tensors.

    Returns
        torch.Tensor, list of int: Padded tensors and original lengths of tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    padded = torch.stack(padded, dim=dim).contiguous()
    return padded, lengths


class SequenceEncoder(Encoder):

    def batch_encode(self, batch, *args, dim=0, **kwargs):
        """
        Returns
            torch.Tensor: Encoded and padded batch of tensors.
            list of int: Original lengths of tensors.
        """
        return pad_batch([self.encode(object_, *args, **kwargs) for object_ in batch],
                         padding_index=self.padding_index,
                         dim=dim)

    def batch_decode(self, batch, lengths=None, *args, **kwargs):
        """ Returns a :class:`list` decoding of the `batch` of :class:`torch.Tensor`. """
        split = batch.split(1) if torch.is_tensor(batch) else batch
        decoded = []
        for i, sequence in enumerate(split):
            sequence = sequence.squeeze(0) if torch.is_tensor(batch) else sequence
            sequence = sequence[:lengths[i]] if lengths is not None else sequence
            decoded.append(self.decode(sequence, *args, **kwargs))
        return decoded
