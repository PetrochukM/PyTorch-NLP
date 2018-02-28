class TextEncoder(object):

    def __init__(self):
        raise NotImplementedError

    def encode(self, string):
        """ Given a string encode it into a tensor """
        raise NotImplementedError

    def decode(self, tensor):
        """ Given a tensor decode it into a string """
        raise NotImplementedError

    @property
    def vocab_size(self):
        """ Return the size of the vocab used to encode the text """
        return len(self.vocab)

    @property
    def vocab(self):
        """ Return an array of the vocab such that index matches the token in encode """
        return NotImplementedError
