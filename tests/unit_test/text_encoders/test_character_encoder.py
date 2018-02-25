import unittest

from lib.text_encoders import CharacterEncoder
from lib.text_encoders import EOS_INDEX


class TestCharacterEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = CharacterEncoder([
            "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
            'know.', '', 'Groucho Marx',
            "I haven't slept for 10 days... because that would be too long.", '', 'Mitch Hedberg'
        ])

    def test_encode(self):
        encoded = self.encoder.encode('Slept in my pajamas, yesterday')
        self.assertEqual(encoded.size()[0], 31)
        self.assertEqual(encoded[-1], EOS_INDEX)

    def test_decode(self):
        encoded = self.encoder.encode('Slept in my pajamas, yesterday')
        decoded = self.encoder.decode(encoded)
        self.assertEqual('slept in my pajamas, yesterday<eos>', decoded)