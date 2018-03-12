import random

from torch.utils.data.sampler import BatchSampler


class ShuffleBatchSampler(BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.

    The `ShuffleBatchSampler` adds `shuffle` on top of `torch.utils.data.sampler.BatchSampler`.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool, optional): If ``True``, the sampler will drop the last batch if its size
            would be less than ``batch_size``.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.

    Example:
        >>> list(ShuffleBatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(ShuffleBatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
            self,
            sampler,
            batch_size,
            drop_last=True,
            shuffle=True,
    ):
        self.shuffle = shuffle
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        # NOTE: This is not data
        batches = list(super().__iter__())
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)
