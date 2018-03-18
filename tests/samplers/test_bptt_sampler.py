import random

from torchnlp.samplers import BPTTSampler


def test_bptt_sampler_odd():
    sampler = BPTTSampler(range(5), 2)
    assert list(sampler) == [(slice(0, 2), slice(1, 3)), (slice(2, 4), slice(3, 5))]
    assert len(sampler) == 2


def test_bptt_sampler_even():
    sampler = BPTTSampler(range(6), 2)
    assert list(sampler) == [(slice(0, 2), slice(1, 3)), (slice(2, 4), slice(3, 5)), (slice(4, 5),
                                                                                      slice(5, 6))]
    assert len(sampler) == 3


def test_bptt_sampler_length():
    for i in range(1, 1000):
        sampler = BPTTSampler(range(i), random.randint(1, 17))
        assert len(sampler) == len(list(sampler))
