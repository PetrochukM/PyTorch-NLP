import pickle

from torchnlp.samplers import SortedSampler


def test_sorted_sampler():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda r: r[0]
    indexes = list(SortedSampler(data_source, sort_key=sort_key))
    assert len(indexes) == len(data_source)
    for i, j in enumerate(indexes):
        assert i == j


def test_pickleable():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sampler = SortedSampler(data_source)
    pickle.dumps(sampler)
