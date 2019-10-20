from functools import partial

from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s, tokenizer):
    return [w.text for w in tokenizer(s)]


class SpacyEncoder(StaticTokenizerEncoder):
    """ Encodes the text using spaCy's tokenizer.

    **Tokenizer Reference:**
    https://spacy.io/api/tokenizer

    Args:
        **args: Arguments passed onto ``StaticTokenizerEncoder.__init__``.
        language (string, optional): Language to use for parsing. Accepted values
            are 'en', 'de', 'es', 'pt', 'fr', 'it', 'nl' and 'xx'.
            For details see https://spacy.io/models/#available-models
        **kwargs: Keyword arguments passed onto ``StaticTokenizerEncoder.__init__``.
    Example:

        >>> encoder = SpacyEncoder(["This ain't funny.", "Don't?"])
        >>> encoder.encode("This ain't funny.")
        tensor([5, 6, 7, 8, 9])
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', 'ai', "n't", 'funny', '.', 'Do', '?']
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ai n't funny ."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('``SpacyEncoder`` does not take keyword argument ``tokenize``.')

        try:
            import spacy
        except ImportError:
            print("Please install spaCy: " "`pip install spacy`")
            raise

        # Use English as default when no language was specified
        language = kwargs.get('language', 'en')

        # All languages supported by spaCy can be found here:
        #   https://spacy.io/models/#available-models
        supported_languages = ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl', 'xx']

        if language in supported_languages:
            # Load the spaCy language model if it has been installed
            try:
                self.spacy = spacy.load(language, disable=['parser', 'tagger', 'ner'])
            except OSError:
                raise ValueError(("Language '{0}' not found. Install using "
                                  "spaCy: `python -m spacy download {0}`").format(language))
        else:
            raise ValueError(
                ("No tokenizer available for language '%s'. " + "Currently supported are %s") %
                (language, supported_languages))

        super().__init__(*args, tokenize=partial(_tokenize, tokenizer=self.spacy), **kwargs)

    def batch_encode(self, sequences):
        # Batch tokenization is handled by ``self.spacy.pipe``
        original = self.tokenize
        self.tokenize = lambda sequence: [token.text for token in sequence]
        return_ = super().batch_encode(self.spacy.pipe(sequences))
        self.tokenize = original
        return return_
