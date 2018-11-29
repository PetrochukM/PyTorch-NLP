import pickle

import pytest

from torchnlp.encoders import LabelEncoder
from torchnlp.encoders import UNKNOWN_TOKEN


@pytest.fixture
def encoder():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    return LabelEncoder(sample)


def test_label_encoder_scalar(encoder):
    input_ = 'symbols/namesake/named_after'
    output = encoder.encode(input_)[0]
    assert encoder.decode(output) == UNKNOWN_TOKEN


def test_label_encoder_unknown(encoder):
    input_ = 'symbols/namesake/named_after'
    output = encoder.encode(input_)
    assert len(output) == 1
    assert encoder.decode(output) == UNKNOWN_TOKEN


def test_label_encoder_known():
    input_ = 'symbols/namesake/named_after'
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    sample.append(input_)
    encoder = LabelEncoder(sample)
    output = encoder.encode(input_)
    assert len(output) == 1
    assert encoder.decode(output) == input_


def test_label_encoder_sequence(encoder):
    input_ = ['symbols/namesake/named_after', 'people/deceased_person/place_of_death']
    output = encoder.encode(input_)
    assert len(output) == 2
    assert encoder.decode(output) == [UNKNOWN_TOKEN, 'people/deceased_person/place_of_death']


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
