import pickle

from torch.utils.data.sampler import SequentialSampler

from torchnlp.samplers import DistributedSampler


def test_distributed_batch_sampler():
    sampler = SequentialSampler(list(range(15)))

    distributed_sampler = DistributedSampler(sampler, num_replicas=3, rank=0)
    assert list(distributed_sampler) == [0, 3, 6, 9, 12]

    distributed_sampler = DistributedSampler(sampler, num_replicas=3, rank=1)
    assert list(distributed_sampler) == [1, 4, 7, 10, 13]

    distributed_sampler = DistributedSampler(sampler, num_replicas=3, rank=2)
    assert list(distributed_sampler) == [2, 5, 8, 11, 14]


def test_pickleable():
    sampler = SequentialSampler(list(range(15)))
    sampler = DistributedSampler(sampler, num_replicas=3, rank=2)
    pickle.dumps(sampler)
