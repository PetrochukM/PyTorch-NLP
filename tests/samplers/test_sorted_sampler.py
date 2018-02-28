from torchnlp.samplers import SortedSampler


def test_sorted_sampler():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda r: r[0]
    batch_size = 2
    indexes = list(SortedSampler(data_source, sort_key))
    assert len(indexes) == len(data_source)
    for i, j in enumerate(indexes):
        assert i == j
