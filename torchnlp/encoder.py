class Encoder(object):
    """ Base class for a text encoder.
    """

    def __init__(self):  # pragma: no cover
        raise NotImplementedError

    def encode(self, string):  # pragma: no cover
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        raise NotImplementedError

    def batch_encode(self, strings, *args, **kwargs):
        """ Returns a :class:`list` of :class:`torch.LongTensor` encoding of the `text`. """
        return [self.encode(s, *args, **kwargs) for s in strings]

    def decode(self, tensor):  # pragma: no cover
        """ Given a :class:`torch.Tensor`, returns a :class:`str` representing the decoded text.

        Note that, depending on the tokenization method, the decoded version is not guaranteed to be
        the original text.
        """
        raise NotImplementedError

    @property
    def vocab_size(self):
        """ Return the size (:class:`int`) of the vocabulary. """
        return len(self.vocab)

    @property
    def vocab(self):  # pragma: no cover
        """ Returns the vocabulary (:class:`list`) used to encode text. """
        return NotImplementedError
