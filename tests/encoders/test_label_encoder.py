import pickle

import pytest

from torchnlp.encoders import IdentityEncoder
from torchnlp.encoders import UNKNOWN_TOKEN


@pytest.fixture
def encoder():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    return IdentityEncoder(sample)


def test_identity_encoder_unknown(encoder):
    input_ = 'symbols/namesake/named_after'
    output = encoder.encode(input_)
    assert len(output) == 1
    assert encoder.decode(output) == UNKNOWN_TOKEN


def test_identity_encoder_known():
    input_ = 'symbols/namesake/named_after'
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    sample.append(input_)
    encoder = IdentityEncoder(sample)
    output = encoder.encode(input_)
    assert len(output) == 1
    assert encoder.decode(output) == input_


def test_identity_encoder_sequence(encoder):
    input_ = ['symbols/namesake/named_after', 'people/deceased_person/place_of_death']
    output = encoder.encode(input_)
    assert len(output) == 2
    assert encoder.decode(output) == [UNKNOWN_TOKEN, 'people/deceased_person/place_of_death']


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
