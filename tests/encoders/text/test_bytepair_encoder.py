import unittest
import torch
import sys
from torchnlp.encoders.text import BPEEncoder


class TestBPETextTokenizer(unittest.TestCase):

    def setUp(self):
        self.corpus = ['This is a corpus of text that provides a bunch of tokens from which ',
                       'to build a vocabulary. It will be used when strings are encoded ',
                       'with a SubwordTextTokenizer subclass. The encoder was coded by a coder.']

    def test_vocab(self):
        encoder = BPEEncoder(self.corpus, from_filenames=False)

        # test if reserved_tokens were add to index_to_token.
        self.assertEqual('<pad>', encoder.vocab[0])
        self.assertEqual('<unk>', encoder.vocab[1])
        self.assertEqual('</s>', encoder.vocab[2])
        self.assertEqual('<s>', encoder.vocab[3])
        self.assertEqual('<copy>', encoder.vocab[4])

        # test if some high occurrence sub words are in the token.
        self.assertIn('oken@@', encoder.index_to_token)
        self.assertIn('encode@@', encoder.index_to_token)

        expect_vocab_size = 57
        self.assertEqual(expect_vocab_size, encoder.vocab_size)

    def test_encode(self):
        if sys.version_info.minor > 5:
            original = 'This is a coded sentence encoded by the SubwordTextTokenizer.'
            encoder = BPEEncoder(self.corpus, from_filenames=False)

            # excepted encode.
            expect = [5, 6, 6, 7, 56, 32, 43, 1, 14, 1, 34, 42, 47, 32, 41, 36, 14, 17,
                      42, 49, 50, 51, 33, 9, 52, 53, 15, 14, 53, 26, 21, 54, 44, 55, 37]

            encode_lst = encoder.encode(original).numpy().tolist()

            self.assertListEqual(expect, encode_lst)

    def test_decoder(self):
        if sys.version_info.minor > 5:
            encoded = torch.tensor([5, 6, 6, 7, 56, 32, 43, 1, 14, 1, 34, 42, 47, 32,
                                    41, 36, 14, 17, 42, 49, 50, 51, 33, 9, 52, 53, 15,
                                    14, 53, 26, 21, 54, 44, 55, 37])

            encoder = BPEEncoder(self.corpus, from_filenames=False)

            expect = "This is a coded s<unk> t<unk> ce encoded by the SubwordTextTokenizer."

            self.assertEqual(expect, encoder.decode(encoded))

    def test_encode_decode(self):
        original = "This is a coded sentence encoded by the SubwordTextTokenizer."
        expect = "This is a coded s<unk> t<unk> ce encoded by the SubwordTextTokenizer."

        encoder = BPEEncoder(self.corpus, from_filenames=False)

        decode_encode_str = encoder.decode(encoder.encode(original))
        self.assertEqual(expect, decode_encode_str)
