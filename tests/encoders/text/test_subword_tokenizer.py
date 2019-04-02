import unittest
import collections
import pickle
import random
import mock

import six

from torchnlp.encoders.text import subword_text_tokenizer
from torchnlp.encoders.text.subword_text_tokenizer import encode
from torchnlp.encoders.text.subword_text_tokenizer import decode
from torchnlp.encoders.text.subword_text_tokenizer import _escape_token
from torchnlp.encoders.text.subword_text_tokenizer import _unescape_token
from torchnlp.encoders.text.subword_text_tokenizer import _ESCAPE_CHARS
from torchnlp.encoders.text.subword_text_tokenizer import SubwordTextTokenizer


class TestTokenCounts(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx',
            "I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg'
        ]

    def test_token_counts(self):
        token_counts = SubwordTextTokenizer._count_tokens(self.corpus)
        expected = {
            u"'": 2,
            u".": 2,
            u". ": 1,
            u"... ": 1,
            u"Groucho": 1,
            u"Marx": 1,
            u"Mitch": 1,
            u"Hedberg": 1,
            u"I": 3,
            u"in": 2,
            u"my": 2,
            u"pajamas": 2,
        }
        self.assertDictContainsSubset(expected, token_counts)


class EncodeDecodeTest(unittest.TestCase):

    def test_encode(self):
        self.assertListEqual([u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."],
                             encode(u"Dude - that's so cool."))
        self.assertListEqual([u"Łukasz", u"est", u"né", u"en", u"1981", u"."],
                             encode(u"Łukasz est né en 1981."))
        self.assertListEqual([u" ", u"Spaces", u"at", u"the", u"ends", u" "],
                             encode(u" Spaces at the ends "))
        self.assertListEqual([u"802", u".", u"11b"], encode(u"802.11b"))
        self.assertListEqual([u"two", u". \n", u"lines"], encode(u"two. \nlines"))

    def test_decode(self):
        self.assertEqual(u"Dude - that's so cool.",
                         decode([u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]))

    def test_invertibility_on_random_strings(self):
        for _ in range(1000):
            s = u"".join(six.unichr(random.randint(0, 65535)) for _ in range(10))
            self.assertEqual(s, decode(encode(s)))


class EscapeUnescapeTokenTest(unittest.TestCase):

    def test_escape_token(self):
        escaped = _escape_token('Foo! Bar.\nunder_score back\\slash',
                                set('abcdefghijklmnopqrstuvwxyz .\n') | _ESCAPE_CHARS)

        self.assertEqual('\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_', escaped)

    def test_unescape_token(self):
        unescaped = _unescape_token('\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_')

        self.assertEqual('Foo! Bar.\nunder_score back\\slash', unescaped)


class SubwordTextTokenizerTest(unittest.TestCase):

    def test_encode_decode(self):
        corpus = ('This is a corpus of text that provides a bunch of tokens from which '
                  'to build a vocabulary. It will be used when strings are encoded '
                  'with a SubwordTextTokenizer subclass. The encoder was coded by a coder.')
        alphabet = set(corpus) ^ {' '}

        original = 'This is a coded sentence encoded by the SubwordTextTokenizer.'

        encoder = SubwordTextTokenizer.build_to_target_size_from_corpus([corpus, original],
                                                                        target_size=100,
                                                                        min_val=2,
                                                                        max_val=10)

        # Encoding should be reversible.
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        self.assertEqual(original, decoded)

        # The substrings coded and coder are frequent enough in the corpus that
        # they should appear in the vocabulary even though they are substrings
        # of other included strings.
        subtoken_strings = encoded
        self.assertIn('encoded_', subtoken_strings)
        self.assertIn('coded_', subtoken_strings)
        self.assertIn('SubwordTextTokenizer_', encoder._all_subtoken_strings)
        self.assertIn('coder_', encoder._all_subtoken_strings)

        # Every character in the corpus should be in the encoder's alphabet and
        # its subtoken vocabulary.
        self.assertTrue(alphabet.issubset(encoder._alphabet))
        for a in alphabet:
            self.assertIn(a, encoder._all_subtoken_strings)

    def test_unicode(self):
        corpus = 'Cat emoticons. \U0001F638 \U0001F639 \U0001F63A \U0001F63B'
        token_counts = collections.Counter(corpus.split(' '))

        encoder = SubwordTextTokenizer.build_to_target_size_from_token_counts(
            100, token_counts, 2, 10)

        self.assertIn('\U0001F638', encoder._alphabet)
        self.assertIn('\U0001F63B', encoder._all_subtoken_strings)

    def test_small_vocab(self):
        corpus = 'The quick brown fox jumps over the lazy dog'
        token_counts = collections.Counter(corpus.split(' '))
        alphabet = set(corpus) ^ {' '}

        encoder = SubwordTextTokenizer.build_to_target_size_from_token_counts(
            10, token_counts, 2, 10)

        # All vocabulary elements are in the alphabet and subtoken strings even
        # if we requested a smaller vocabulary to assure all expected strings
        # are encodable.
        self.assertTrue(alphabet.issubset(encoder._alphabet))
        for a in alphabet:
            self.assertIn(a, encoder._all_subtoken_strings)

    def test_encodable_when_not_in_alphabet(self):
        corpus = 'the quick brown fox jumps over the lazy dog'
        token_counts = collections.Counter(corpus.split(' '))

        encoder = SubwordTextTokenizer.build_to_target_size_from_token_counts(
            100, token_counts, 2, 10)
        original = 'This has UPPER CASE letters that are out of alphabet'

        # Early versions could have an infinite loop when breaking into subtokens
        # if there was any out-of-alphabet characters in the encoded string.
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)

        self.assertEqual(original, decoded)
        encoded_str = ''.join(encoded)
        self.assertIn('\\84;', encoded_str)

    @mock.patch.object(subword_text_tokenizer, '_ESCAPE_CHARS', new=set('\\_;13579'))
    def test_raises_exception_when_not_encodable(self):
        corpus = 'the quick brown fox jumps over the lazy dog'
        token_counts = collections.Counter(corpus.split(' '))

        # Deliberately exclude some required encoding chars from the alphabet
        # and token list, making some strings unencodable.
        encoder = SubwordTextTokenizer.build_to_target_size_from_token_counts(
            100, token_counts, 2, 10)
        original = 'This has UPPER CASE letters that are out of alphabet'

        # Previously there was a bug which produced an infinite loop in this case.
        with self.assertRaises(AssertionError):
            encoder.encode(original)


def test_is_pickleable():
    tokenizer = SubwordTextTokenizer()
    pickle.dumps(tokenizer)
