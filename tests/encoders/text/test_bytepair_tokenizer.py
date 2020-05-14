import unittest
import pickle

from torchnlp.encoders.text.bpe_text_tokenizer import BPETextTokenizer


class TestBPETextTokenizer(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', 'Groucho Marx',
            "I haven't slept for 10 days... because that would be too long.", 'Mitch Hedberg'
        ]

    def test_pre_tokenizer(self):
        expected = ['One morning I shot an elephant in my pajamas . How he got in my pajamas ,'
                    ' I don &apos;t',
                    'know .',
                    'Groucho Marx',
                    'I haven &apos;t slept for 10 days ... because that would be too long .',
                    'Mitch Hedberg']

        self.assertListEqual(expected, [BPETextTokenizer.pre_tokenize(sen) for sen in self.corpus])

    def test_get_vocabulary(self):
        # tokenizer = BPETextTokenizer('test_bpe', use_moses=True)
        def segment_words(line):
            return BPETextTokenizer._segment_words(line, BPETextTokenizer.pre_tokenize)
        token_counts = BPETextTokenizer.get_vocabulary(self.corpus,
                                                       segment_words, from_filenames=False)
        expected = {
            "&apos;t": 2,
            ".": 3,
            "...": 1,
            "Groucho": 1,
            "Marx": 1,
            "Mitch": 1,
            "Hedberg": 1,
            "I": 3,
            "in": 2,
            "my": 2,
            "know": 1,
            "because": 1,
            "pajamas": 2,
        }
        self.assertDictContainsSubset(expected, token_counts)

    def test_learn_bpe(self):
        tokenizer = BPETextTokenizer('test_bpe')
        tokenizer.build_from_corpus(self.corpus, from_filenames=False)
        expected = {('&', 'apos;t</w>'): 21, ('a', 'pos;t</w>'): 20, ('b', 'e'): 19,
                    ('i', 'n</w>'): 18, ('le', 'p'): 17, ('l', 'e'): 16, ('m', 'y</w>'): 15,
                    ('n', 'g</w>'): 14, ('o', 't</w>'): 13, ('o', 'u'): 12, ('o', 'w</w>'): 11,
                    ('pajama', 's</w>'): 10, ('pajam', 'a'): 9, ('paja', 'm'): 8, ('paj', 'a'): 7,
                    ('pa', 'j'): 6, ('p', 'a'): 5, ('po', 's;t</w>'): 4, ('p', 'o'): 3,
                    ('s;', 't</w>'): 2, ('s', ';'): 1, ('h', 'a'): 0}
        self.assertDictEqual(expected, tokenizer.bpe.bpe_codes)

    def test_encode_decode(self):
        corpus = ['This is a corpus of text that provides a bunch of tokens from which ',
                  'to build a vocabulary. It will be used when strings are encoded ',
                  'with a SubwordTextTokenizer subclass. The encoder was coded by a coder.']

        original = 'This is a coded sentence encoded by the SubwordTextTokenizer.'

        tokenizer = BPETextTokenizer('test_bpe')
        tokenizer.build_from_corpus(corpus, from_filenames=False)

        # Encoding should be reversible.
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(original, decoded)

        # The substrings coded@@ and en@@ are frequent enough in the corpus that
        # they should appear in the vocabulary even though they are substrings
        # of other included strings.
        subtoken_strings = encoded
        self.assertIn('en@@', subtoken_strings)
        self.assertIn('code@@', subtoken_strings)

    def test_build_vocab(self):
        tokenizer = BPETextTokenizer('test_bpe')
        tokenizer.build_from_corpus(self.corpus, from_filenames=False)

        # test the all item in vocab.
        expect = {'O@@': 1, 'n@@': 4, 'e': 4, 'm@@': 1, 'o@@': 5, 'r@@': 4, 'i@@': 2,
                  'ng': 2, 'I': 3, 's@@': 3, 'h@@': 3, 'ot': 2, 'a@@': 4, 'n': 3,
                  'e@@': 3, 'lep@@': 2, 'ha@@': 3, 't': 3, 'in': 2, 'my': 2,
                  'pajamas': 2, '.': 4, 'H@@': 2, 'ow': 2, 'g@@': 1, ',': 1, 'd@@': 3,
                  '&apos;t': 2, 'k@@': 1, 'G@@': 1, 'ou@@': 2, 'c@@': 3, 'o': 2,
                  'M@@': 2, 'x': 1, 'v@@': 1, 'f@@': 1, 'r': 1, '1@@': 1, '0': 1,
                  'y@@': 1, 's': 1, '.@@': 2, 'be@@': 2, 'u@@': 1, 't@@': 3,
                  'w@@': 1, 'l@@': 2, 'd': 1, 'b@@': 1, 'h': 1, 'g': 1}

        self.assertDictEqual(expect, tokenizer.vocab)


def test_is_pickleable():
    tokenizer = BPETextTokenizer('test_bpe')
    pickle.dumps(tokenizer)
