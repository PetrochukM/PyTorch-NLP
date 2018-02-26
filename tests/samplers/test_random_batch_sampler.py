from lib.samplers import RandomBatchSampler

from lib.samplers import SortedSampler


def test_random_batch_sampler():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(RandomBatchSampler(SortedSampler(data_source, sort_key), batch_size, False))
    assert len(batches) == 3


def test_random_batch_sampler_drop_last():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(
        RandomBatchSampler(SortedSampler(data_source, sort_key), batch_size, drop_last=True))
    assert len(batches) == 2
