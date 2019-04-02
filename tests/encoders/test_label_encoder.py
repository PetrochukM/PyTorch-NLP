import pickle

import pytest
import torch

from torchnlp.encoders import LabelEncoder
from torchnlp.encoders.label_encoder import DEFAULT_UNKNOWN_TOKEN


@pytest.fixture
def label_encoder():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    return LabelEncoder(sample)


def test_label_encoder_no_reserved():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    label_encoder = LabelEncoder(sample, reserved_labels=[], unknown_index=None)

    label_encoder.encode('people/deceased_person/place_of_death')

    # No ``unknown_index`` defined causes ``RuntimeError`` if an unknown label is used.
    with pytest.raises(RuntimeError):
        label_encoder.encode('symbols/namesake/named_after')


def test_label_encoder_enforce_reversible(label_encoder):
    label_encoder.enforce_reversible()

    with pytest.raises(ValueError):
        label_encoder.encode('symbols/namesake/named_after')

    with pytest.raises(IndexError):
        label_encoder.decode(torch.tensor(label_encoder.vocab_size))


def test_label_encoder_batch_encoding(label_encoder):
    encoded = label_encoder.batch_encode(label_encoder.vocab)
    assert torch.equal(encoded, torch.arange(label_encoder.vocab_size).view(-1))


def test_label_encoder_batch_decoding(label_encoder):
    assert label_encoder.vocab == label_encoder.batch_decode(torch.arange(label_encoder.vocab_size))


def test_label_encoder_vocab(label_encoder):
    assert len(label_encoder.vocab) == 3
    assert len(label_encoder.vocab) == label_encoder.vocab_size


def test_label_encoder_unknown(label_encoder):
    input_ = 'symbols/namesake/named_after'
    output = label_encoder.encode(input_)
    assert label_encoder.decode(output) == DEFAULT_UNKNOWN_TOKEN


def test_label_encoder_known(label_encoder):
    input_ = 'symbols/namesake/named_after'
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    sample.append(input_)
    label_encoder = LabelEncoder(sample)
    output = label_encoder.encode(input_)
    assert label_encoder.decode(output) == input_


def test_label_encoder_is_pickleable(label_encoder):
    pickle.dumps(label_encoder)
