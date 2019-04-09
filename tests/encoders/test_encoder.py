from torchnlp.encoders import Encoder


def test_encoder():
    encoder = Encoder(enforce_reversible=True)
    encoder.encode('this is a test')
    encoder.decode('this is a test')
