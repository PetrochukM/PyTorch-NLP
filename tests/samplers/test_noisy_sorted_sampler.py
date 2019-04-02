import pickle

from torchnlp.samplers import NoisySortedSampler


def test_noisy_sorted_sampler():
    data_source = [1, 2, 3, 4, 5, 6]
    indexes = list(NoisySortedSampler(data_source))
    assert len(indexes) == len(data_source)


def test_noisy_sorted_sampler_sorted():
    data_source = [1, 2, 3, 4, 5, 6]
    indexes = list(NoisySortedSampler(data_source, sort_key_noise=0.0))
    assert len(indexes) == len(data_source)
    for i, j in enumerate(indexes):
        assert i == j


def test_noisy_sorted_sampler_sort_key_noise():
    data_source = [2, 6, 10]
    # `sort_key_noise` does not affect values 2, 6, 10
    indexes = list(NoisySortedSampler(data_source, sort_key_noise=0.25))
    for i, j in enumerate(indexes):
        assert i == j


def test_pickleable():
    data_source = [1, 2, 3, 4, 5, 6]
    sampler = NoisySortedSampler(data_source)
    pickle.dumps(sampler)
