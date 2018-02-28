from torchnlp.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class WordEncoder(StaticTokenizerEncoder):
    """ Split a string by spaces """

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
