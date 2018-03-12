import spacy
from spacy.lang.en import English

from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder

# Use the SpacyEncoder by downloading en_core_web_sm via: `python -m spacy download en_core_web_sm`
_MODEL = spacy.load('en_core_web_sm')
tokenizer = English().Defaults.create_tokenizer(_MODEL)


class SpacyEncoder(StaticTokenizerEncoder):
    """ Encodes the text using the Spacy `en_core_web_sm` tokenizer.

    Tokenization Algorithm Reference:
    https://spacy.io/api/tokenizer

    Args:
        sample (list of strings): Sample of data to build dictionary on
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

        super().__init__(*args, **kwargs, tokenize=lambda s: [w.text for w in tokenizer(s)])
