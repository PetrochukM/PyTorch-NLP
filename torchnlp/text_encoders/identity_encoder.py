from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class IdentityEncoder(StaticTokenizerEncoder):
    """ No tokenization for example: 'Hi There' => ['Hi There'] """

    def __init__(self, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('IdentityEncoder defines a identity tokenize s -> [s]')
        if 'append_eos' not in kwargs:
            kwargs['append_eos'] = False  # Default to not appending EOS
        super().__init__(*args, **kwargs, tokenize=(lambda s: [s]))

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ''.join(tokens)
