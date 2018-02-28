from torchnlp.text_encoders.word_encoder import WordEncoder


class MosesEncoder(WordEncoder):
    """ Use Moses to encode. """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('MosesEncoder defines a tokenize callable Moses')

        import nltk

        # Required for moses
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')

        from nltk.tokenize.moses import MosesTokenizer
        from nltk.tokenize.moses import MosesDetokenizer

        self.detokenizer = MosesDetokenizer()

        super().__init__(*args, **kwargs, tokenize=MosesTokenizer().tokenize)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.detokenizer.detokenize(tokens, return_str=True)
