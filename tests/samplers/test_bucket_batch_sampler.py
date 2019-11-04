import pickle

from torch.utils.data.sampler import SequentialSampler

from torchnlp.random import fork_rng_wrap
from torchnlp.samplers import BucketBatchSampler


def test_bucket_batch_sampler_length():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda i: len(data_source[i])
    batch_size = 2
    sampler = SequentialSampler(data_source)
    batch_sampler = BucketBatchSampler(
        sampler,
        batch_size=batch_size,
        sort_key=sort_key,
        drop_last=False,
        bucket_size_multiplier=2)
    batches = list(batch_sampler)
    assert len(batches) == 3
    assert len(batch_sampler) == 3


def test_bucket_batch_sampler_uneven_length():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda i: len(data_source[i])
    batch_size = 2
    sampler = SequentialSampler(data_source)
    batch_sampler = BucketBatchSampler(
        sampler, batch_size, sort_key=sort_key, drop_last=False, bucket_size_multiplier=2)
    batches = list(batch_sampler)
    assert len(batches) == 3
    assert len(batch_sampler) == 3
    batch_sampler = BucketBatchSampler(
        sampler, batch_size, sort_key=sort_key, drop_last=True, bucket_size_multiplier=2)
    batches = list(batch_sampler)
    assert len(batches) == 2
    assert len(batch_sampler) == 2


def test_bucket_batch_sampler_sorted():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda i: data_source[i]
    batch_size = len(data_source)
    sampler = SequentialSampler(data_source)
    batches = list(
        BucketBatchSampler(
            sampler, batch_size, sort_key=sort_key, drop_last=False, bucket_size_multiplier=1))
    for i, batch in enumerate(batches):
        assert batch[0] == i


@fork_rng_wrap(seed=123)
def test_bucket_batch_sampler():
    sampler = SequentialSampler(list(range(10)))
    batch_sampler = BucketBatchSampler(
        sampler, batch_size=3, drop_last=False, bucket_size_multiplier=2)
    assert len(batch_sampler) == 4
    assert list(batch_sampler) == [[0, 1, 2], [3, 4, 5], [9], [6, 7, 8]]


def test_bucket_batch_sampler__drop_last():
    sampler = SequentialSampler(list(range(10)))
    batch_sampler = BucketBatchSampler(
        sampler, batch_size=3, drop_last=True, bucket_size_multiplier=2)
    assert len(batch_sampler) == 3
    assert len(list(iter(batch_sampler))) == 3


def test_pickleable():
    sampler = SequentialSampler(list(range(10)))
    batch_sampler = BucketBatchSampler(
        sampler, batch_size=2, drop_last=False, bucket_size_multiplier=2)
    pickle.dumps(batch_sampler)
