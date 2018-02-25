import spacy

from lib.text_encoders.word_encoder import WordEncoder

NLP = spacy.load('en_core_web_sm')


def spacy_tokenize(s):
    doc = NLP(s, disable=['parser', 'tagger', 'ner'])
    return [w.text for w in doc]


class SpacyEncoder(WordEncoder):
    """ Use Moses to encode. """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('TreebankEncoder defines a tokenize callable Moses')

        super().__init__(*args, **kwargs, tokenize=spacy_tokenize)
