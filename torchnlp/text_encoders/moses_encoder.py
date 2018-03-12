from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class MosesEncoder(StaticTokenizerEncoder):
    """ Encodes the text using the Moses tokenizer.

    Tokenization Algorithm Reference:
    http://www.nltk.org/_modules/nltk/tokenize/moses.html

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:

        >>> encoder = MosesEncoder(["This ain't funny.", "Don't?"])
        >>> encoder.encode("This ain't funny.")
         5
         6
         7
         8
         9
        [torch.LongTensor of size 5]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', 'ain', '&apos;t', 'funny', '.', 'Don',
        '?']
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('MosesEncoder defines a tokenize callable Moses')

        try:
            import nltk

            # Required for moses
            nltk.download('perluniprops')
            nltk.download('nonbreaking_prefixes')

            from nltk.tokenize.moses import MosesTokenizer
            from nltk.tokenize.moses import MosesDetokenizer
        except ImportError:
            print("Please install NLTK. " "See the docs at http://nltk.org for more information.")
            raise

        self.detokenizer = MosesDetokenizer()

        super().__init__(*args, **kwargs, tokenize=MosesTokenizer().tokenize)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.detokenizer.detokenize(tokens, return_str=True)
