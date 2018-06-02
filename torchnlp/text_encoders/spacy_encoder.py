from functools import partial

import torch

from torchnlp.text_encoders.reserved_tokens import EOS_INDEX
from torchnlp.text_encoders.reserved_tokens import UNKNOWN_INDEX
from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s, tokenizer):
    return [w.text for w in tokenizer(s)]


class SpacyEncoder(StaticTokenizerEncoder):
    """ Encodes the text using spaCy's tokenizer.

    **Tokenizer Reference:**
    https://spacy.io/api/tokenizer

    Args:
        sample (list of strings): Sample of data to build dictionary on
        language (string, optional): Language to use for parsing. Accepted values
            are 'en', 'de', 'es', 'pt', 'fr', 'it', 'nl' and 'xx'.
            For details see https://spacy.io/models/#available-models
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:

        >>> encoder = SpacyEncoder(["This ain't funny.", "Don't?"])
        >>> encoder.encode("This ain't funny.")
         5
         6
         7
         8
         9
        [torch.LongTensor of size 5]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', 'ai', "n't", 'funny', '.', 'Do', '?']
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ai n't funny ."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('SpacyEncoder defines a tokenize callable.')

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
                raise ValueError(("Language '{0}' not found. Install using " +
                                  "spaCy: `python -m spacy download {0}`").format(language))
        else:
            raise ValueError(
                ("No tokenizer available for language '%s'. " + "Currently supported are %s") %
                (language, supported_languages))

        super().__init__(*args, tokenize=partial(_tokenize, tokenizer=self.spacy), **kwargs)

    def batch_encode(self, texts, eos_index=EOS_INDEX, unknown_index=UNKNOWN_INDEX):
        return_ = []
        for tokens in self.spacy.pipe(texts, n_threads=-1):
            text = [token.text for token in tokens]
            vector = [self.stoi.get(token, unknown_index) for token in text]
            if self.append_eos:
                vector.append(eos_index)
            return_.append(torch.LongTensor(vector))
        return return_
