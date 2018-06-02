import pickle

import torch
from torchnlp.samplers import BucketBatchSampler


def test_bucket_batch_sampler_length():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda r: len(r)
    batch_size = 2
    sampler = BucketBatchSampler(
        data_source, batch_size, sort_key=sort_key, drop_last=False, bucket_size_multiplier=2)
    batches = list(sampler)
    assert len(batches) == 3
    assert len(sampler) == 3


def test_bucket_batch_sampler_uneven_length():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda r: len(r)
    batch_size = 2
    sampler = BucketBatchSampler(
        data_source, batch_size, sort_key=sort_key, drop_last=False, bucket_size_multiplier=2)
    batches = list(sampler)
    assert len(batches) == 3
    assert len(sampler) == 3
    sampler = BucketBatchSampler(
        data_source, batch_size, sort_key=sort_key, drop_last=True, bucket_size_multiplier=2)
    batches = list(sampler)
    assert len(batches) == 2
    assert len(sampler) == 2


def test_bucket_batch_sampler_last_batch_first():
    data_source = [torch.tensor([j for j in range(i)]) for i in range(100)]
    sort_key = lambda r: len(r)
    batch_size = 1
    batches = list(
        BucketBatchSampler(
            data_source, batch_size, sort_key=sort_key, drop_last=False, bucket_size_multiplier=2))
    # Largest batch (4) is in first batch
    assert 99 == batches[0][0]


def test_bucket_batch_sampler_sorted():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda r: r[0]
    batch_size = len(data_source)
    batches = list(
        BucketBatchSampler(
            data_source,
            batch_size,
            sort_key=sort_key,
            drop_last=False,
            biggest_batches_first=None,
            bucket_size_multiplier=1))
    # Largest batch (4) is in first batch
    for i, batch in enumerate(batches):
        assert batch[0] == i


def test_pickleable():
    data_source = [[1], [2], [3], [4], [5]]
    sampler = BucketBatchSampler(data_source, batch_size=2, drop_last=False)
    pickle.dumps(sampler)
