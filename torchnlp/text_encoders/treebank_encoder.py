from torchnlp.text_encoders.word_encoder import WordEncoder


class TreebankEncoder(WordEncoder):
    """ Use Moses to encode. """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('TreebankEncoder defines a tokenize callable TreebankWordTokenizer')

        import nltk

        # Required for moses
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')

        from nltk.tokenize.treebank import TreebankWordTokenizer
        from nltk.tokenize.treebank import TreebankWordDetokenizer

        self.detokenizer = TreebankWordDetokenizer()

        super().__init__(*args, **kwargs, tokenize=TreebankWordTokenizer().tokenize)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.detokenizer.detokenize(tokens)
