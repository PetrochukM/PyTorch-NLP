import random

from torch.utils.data.sampler import BatchSampler


class RandomBatchSampler(BatchSampler):

    def __iter__(self):
        batches = list(super().__iter__())
        random.shuffle(batches)
        return iter(batches)
