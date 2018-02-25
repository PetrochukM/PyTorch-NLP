

from lib.text_encoders.word_encoder import WordEncoder


class TreebankEncoder(WordEncoder):
    """ Use Moses to encode. """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('TreebankEncoder defines a tokenize callable Moses')

        import nltk

        # Required for moses
        nltk.download('perluniprops')
        nltk.download('nonbreaking_prefixes')

        from nltk.tokenize.treebank import TreebankWordTokenizer

        super().__init__(*args, **kwargs, tokenize=TreebankWordTokenizer().tokenize)
