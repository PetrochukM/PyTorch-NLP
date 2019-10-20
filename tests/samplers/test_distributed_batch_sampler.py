import pickle

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler

from torchnlp.samplers import DistributedBatchSampler


def test_distributed_batch_sampler():
    sampler = SequentialSampler(list(range(15)))
    batch_sampler = BatchSampler(sampler, 10, False)

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=0)
    assert list(distributed_sampler) == [[0, 4, 8], [10, 14]]
    assert len(distributed_sampler) == 2

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=1)
    assert list(distributed_sampler) == [[1, 5, 9], [11]]
    assert len(distributed_sampler) == 2

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=2)
    assert list(distributed_sampler) == [[2, 6], [12]]
    assert len(distributed_sampler) == 2

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=3)
    assert list(distributed_sampler) == [[3, 7], [13]]
    assert len(distributed_sampler) == 2


def test_pickleable():
    sampler = SequentialSampler(list(range(15)))
    batch_sampler = BatchSampler(sampler, 10, False)
    sampler = DistributedBatchSampler(batch_sampler, num_replicas=1, rank=0)
    pickle.dumps(sampler)
