class TextEncoder(object):
    """ Base class for a text encoder.
    """

    def __init__(self):
        raise NotImplementedError

    def encode(self, string):
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        raise NotImplementedError

    def decode(self, tensor):
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
    def vocab(self):
        """ Returns the vocabulary (:class:`list`) used to encode text. """
        return NotImplementedError
