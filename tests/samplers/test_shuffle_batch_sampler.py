import pickle

from torchnlp.samplers import ShuffleBatchSampler

from torchnlp.samplers import SortedSampler


def test_shuffle_batch_sampler():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(
        ShuffleBatchSampler(SortedSampler(data_source, sort_key=sort_key), batch_size, False))
    assert len(batches) == 3


def test_shuffle_batch_sampler_drop_last():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(
        ShuffleBatchSampler(
            SortedSampler(data_source, sort_key=sort_key), batch_size, drop_last=True))
    assert len(batches) == 2


def test_pickleable():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sampler = ShuffleBatchSampler(SortedSampler(data_source), batch_size=2, drop_last=False)
    pickle.dumps(sampler)
