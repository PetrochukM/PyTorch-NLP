import spacy
from spacy.lang.en import English

from torchnlp.text_encoders.word_encoder import WordEncoder

# Use the SpacyEncoder by downloading en_core_web_sm via: `python -m spacy download en_core_web_sm`
_MODEL = spacy.load('en_core_web_sm')
tokenizer = English().Defaults.create_tokenizer(_MODEL)


class SpacyEncoder(WordEncoder):
    """ Use Moses to encode. """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('SpacyEncoder defines a tokenize callable.')

        super().__init__(*args, **kwargs, tokenize=lambda s: [w.text for w in tokenizer(s)])
