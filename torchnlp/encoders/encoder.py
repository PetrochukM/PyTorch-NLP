import torch


class Encoder(object):
    """ Base class for a encoder.
    """

    def __init__(self):  # pragma: no cover
        raise NotImplementedError

    def enforce_reversible(self):
        """ Updates the specification of ``Encoder.encode`` and ``Encoder.decode`` to enforce
        reversibility.

        Formally, reversible means: ``Encoder.decode(Encoder.encode(object_)) == object_``

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
        """ Encodes an object to a :class:`torch.Tensor`.

        Args:
            object_ (object): Object to encode.

        Returns:
            torch.Tensor: Encoding of the object.
        """
        raise NotImplementedError

    def batch_encode(self, batch, *args, **kwargs):
        """
        Args:
            batch (list): Batch of objects to encode.
            *args: Arguments passed to ``encode``.
            **kwargs: Key word arguments passed to ``encode``.

        Returns:
            list: Batch of encoded objects.
        """
        return torch.tensor([self.encode(object_, *args, **kwargs).tolist() for object_ in batch])

    def decode(self, tensor):  # pragma: no cover
        """ Decodes a :class:`torch.Tensor` to a object.

        Args:
            tensor (torch.Tensor): Tensor to decode.

        Returns:
            object: Object decoded from tensor.
        """
        raise NotImplementedError

    def batch_decode(self, batch, *args, **kwargs):
        """
        Args:
            batch (list of :class:`torch.Tensor`): Batch of encoded objects.
            *args: Arguments passed to ``decode``.
            **kwargs: Key word arguments passed to ``decode``.

        Returns:
            list: Batch of decoded objects.
        """
        iterator = ([split.squeeze(0) for split in batch.split(1)]
                    if torch.is_tensor(batch) else batch)
        return [self.decode(tensor, *args, **kwargs) for tensor in iterator]
