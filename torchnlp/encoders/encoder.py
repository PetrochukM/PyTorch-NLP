class Encoder(object):
    """
    Base class for a encoder employing an identity function.

    Args:
        enforce_reversible (bool, optional): Check for reversibility on ``Encoder.encode`` and
          ``Encoder.decode``. Formally, reversible means:
          ``Encoder.decode(Encoder.encode(object_)) == object_``.
    """

    def __init__(self, enforce_reversible=False):
        self.enforce_reversible = enforce_reversible

    def encode(self, object_):
        """ Encodes an object.

        Args:
            object_ (object): Object to encode.

        Returns:
            object: Encoding of the object.
        """
        if self.enforce_reversible:
            self.enforce_reversible = False
            encoded_decoded = self.decode(self.encode(object_))
            self.enforce_reversible = True
            if encoded_decoded != object_:
                raise ValueError('Encoding is not reversible for "%s"' % object_)

        return object_

    def batch_encode(self, iterator, *args, **kwargs):
        """
        Args:
            batch (list): Batch of objects to encode.
            *args: Arguments passed to ``encode``.
            **kwargs: Keyword arguments passed to ``encode``.

        Returns:
            list: Batch of encoded objects.
        """
        return [self.encode(object_, *args, **kwargs) for object_ in iterator]

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
            if decoded_encoded != encoded:
                raise ValueError('Decoding is not reversible for "%s"' % encoded)

        return encoded

    def batch_decode(self, iterator, *args, **kwargs):
        """
        Args:
            iterator (list): Batch of encoded objects.
            *args: Arguments passed to ``decode``.
            **kwargs: Keyword arguments passed to ``decode``.

        Returns:
            list: Batch of decoded objects.
        """
        return [self.decode(encoded, *args, **kwargs) for encoded in iterator]
