import unittest

from lib.text_encoders import WordEncoder
from lib.text_encoders import UNKNOWN_INDEX
from lib.text_encoders import EOS_INDEX


class TestWordEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = WordEncoder([
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx',
            "I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg'
        ])

    def test_encode(self):
        encoded = self.encoder.encode('Slept in my pajamas, yesterday')
        self.assertEqual(encoded.size()[0], 6)
        self.assertEqual(encoded[4], UNKNOWN_INDEX)
        self.assertEqual(encoded[5], EOS_INDEX)

    def test_decode(self):
        encoded = self.encoder.encode('Slept in my pajamas, yesterday')
        decoded = self.encoder.decode(encoded)
        self.assertEqual('slept in my pajamas, <unk> <eos>', decoded)
