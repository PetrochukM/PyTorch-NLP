import unittest

from lib.text_encoders import SubwordEncoder
from lib.text_encoders import EOS_INDEX


class TestSubwordEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder_min_val = SubwordEncoder(
            [
                "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
                'know.', '', 'Groucho Marx',
                "I haven't slept for 10 days... because that would be too long.", '',
                'Mitch Hedberg'
            ],
            min_val=2)
        self.encoder_target_size = SubwordEncoder(
            [
                "One morning I shot an elephant in my pajamas. How he got in my pajamas, I don't",
                'know.', '', 'Groucho Marx',
                "I haven't slept for 10 days... because that would be too long.", '',
                'Mitch Hedberg'
            ],
            target_size=100,
            min_val=2,
            max_val=6)

    def test_encode_target_size(self):
        encoded = self.encoder_target_size.encode('Slept in my pajamas, yesterday')
        self.assertEqual(encoded[-1], EOS_INDEX)

    def test_encode_min_val(self):
        encoded = self.encoder_min_val.encode('Slept in my pajamas, yesterday')
        self.assertEqual(encoded[-1], EOS_INDEX)

    def test_decode(self):
        encoded = self.encoder_min_val.encode('Slept in my pajamas, yesterday')
        decoded = self.encoder_min_val.decode(encoded)
        self.assertEqual('slept in my pajamas, yesterday<eos>', decoded)
