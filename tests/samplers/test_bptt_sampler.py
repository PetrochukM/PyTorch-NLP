import pickle
import random

import pytest

from torchnlp.samplers import BPTTSampler


@pytest.fixture
def sampler():
    return BPTTSampler(range(5), 2)


def test_bptt_sampler_odd(sampler):
    assert list(sampler) == [slice(0, 2), slice(2, 4)]
    assert len(sampler) == 2


def test_bptt_sampler_even():
    sampler = BPTTSampler(range(6), 2, type_='target')
    assert list(sampler) == [slice(1, 3), slice(3, 5), slice(5, 6)]
    assert len(sampler) == 3


def test_bptt_sampler_length():
    for i in range(1, 1000):
        sampler = BPTTSampler(range(i), random.randint(1, 17))
        assert len(sampler) == len(list(sampler))


def test_is_pickleable(sampler):
    pickle.dumps(sampler)
