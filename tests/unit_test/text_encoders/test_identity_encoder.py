import unittest

from lib.text_encoders import IdentityEncoder


class TestCharacterEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = IdentityEncoder(
            ['up', 'down', 'left', 'right', 'left', 'right', 'select', 'start'])

    def test_encode(self):
        encoded = self.encoder.encode('up')
        self.assertEqual(encoded.size()[0], 1)

    def test_decode(self):
        encoded = self.encoder.encode('up')
        decoded = self.encoder.decode(encoded)
        self.assertEqual('up', decoded)
