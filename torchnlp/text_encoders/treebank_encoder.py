from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class TreebankEncoder(StaticTokenizerEncoder):
    """ Encodes the text using the Treebank tokenizer.

    Tokenization Algorithm Reference:
    http://www.nltk.org/_modules/nltk/tokenize/treebank.html

    Args:
        sample (list of strings): Sample of data to build dictionary on
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.

    Example:

        >>> encoder = TreebankEncoder(["This ain't funny.", "Don't?"])
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
        "This ain't funny."

    """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('TreebankEncoder defines a tokenize callable TreebankWordTokenizer')

        try:
            import nltk

            # Required for moses
            nltk.download('perluniprops')
            nltk.download('nonbreaking_prefixes')

            from nltk.tokenize.treebank import TreebankWordTokenizer
            from nltk.tokenize.treebank import TreebankWordDetokenizer
        except ImportError:
            print("Please install NLTK. " "See the docs at http://nltk.org for more information.")
            raise

        self.detokenizer = TreebankWordDetokenizer()

        super().__init__(*args, **kwargs, tokenize=TreebankWordTokenizer().tokenize)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return self.detokenizer.detokenize(tokens)
