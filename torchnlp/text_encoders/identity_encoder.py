from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder

# TODO: Already tokenized text should work...


class IdentityEncoder(StaticTokenizerEncoder):
    """ No tokenization for example: 'Hi There' => ['Hi There'] """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('IdentityEncoder defines a identity tokenization')
        if 'append_eos' not in kwargs:
            kwargs['append_eos'] = False  # Default to not appending EOS
        super().__init__(*args, **kwargs, tokenize=(lambda s: s if isinstance(s, list) else [s]))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        if len(tokens) == 1:
            return tokens[0]
        else:
            return tokens
