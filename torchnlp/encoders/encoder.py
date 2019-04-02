import torch


class Encoder(object):
    """ Base class for a encoder.
    """

    def __init__(self):  # pragma: no cover
        raise NotImplementedError

    def enforce_reversible(self):
        """ Updates the specification of `self.encode` and `self.decode` to enforce reversibility.

        Formally, reversible means: ``self.decode(self.encode(object_)) == object_``

        Example:
            >>> encoder = Encoder().enforce_reversible()  # doctest: +SKIP
        """
        last_encode = self.encode
        last_decode = self.decode

        def _encode(object_, *args, **kwargs):
            encoded = last_encode(object_, *args, **kwargs)
            if last_decode(encoded) != object_:
                raise ValueError('Encoding is not reversible for "%s"', object_)
            return encoded

        self.encode = _encode

        def _decode(tensor, *args, **kwargs):
            decoded = last_decode(tensor, *args, **kwargs)
            if torch.equals(last_encode(decoded), tensor):
                raise ValueError('Decode is not reversible for "%s"', tensor)
            return decoded

        self.decode = _decode
        return self

    def encode(self, object_):  # pragma: no cover
        """ Returns a :class:`torch.Tensor` encoding of the `object`. """
        raise NotImplementedError

    def batch_encode(self, batch, *args, **kwargs):
        """ Returns a :class:`torch.Tensor` encoding of the `batch` of `object_`s. """
        return torch.tensor([self.encode(object_, *args, **kwargs).tolist() for object_ in batch])

    def decode(self, tensor):  # pragma: no cover
        """ Given a :class:`torch.Tensor` returns the decoded tensor.

        Note that, depending on the tokenization method, the decoded version is not guaranteed to be
        the same as the original.
        """
        raise NotImplementedError

    def batch_decode(self, batch, *args, **kwargs):
        """ Returns a :class:`list` decoding of the `batch` of :class:`torch.Tensor`. """
        iterator = ([split.squeeze(0) for split in batch.split(1)]
                    if torch.is_tensor(batch) else batch)
        return [self.decode(tensor, *args, **kwargs) for tensor in iterator]
