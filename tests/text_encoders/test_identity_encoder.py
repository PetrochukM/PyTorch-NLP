from torchnlp.text_encoders import IdentityEncoder
from torchnlp.text_encoders import UNKNOWN_TOKEN


def test_delimiter_encoder_unknown():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    encoder = IdentityEncoder(sample)
    input_ = 'symbols/namesake/named_after'
    output = encoder.encode(input_)
    assert len(output) == 1
    assert encoder.decode(output) == UNKNOWN_TOKEN


def test_delimiter_encoder_known():
    input_ = 'symbols/namesake/named_after'
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    sample.append(input_)
    encoder = IdentityEncoder(sample)
    output = encoder.encode(input_)
    assert len(output) == 1
    assert encoder.decode(output) == input_
