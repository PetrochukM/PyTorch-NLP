import pickle

from torch.utils.data.sampler import SequentialSampler

from torchnlp.random import fork_rng_wrap
from torchnlp.samplers import BucketBatchSampler


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
