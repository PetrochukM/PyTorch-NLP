from torch.utils.data.sampler import Sampler

import torch


class DistributedSampler(Sampler):
    """ Iterable wrapper that distributes data across multiple workers.

    Args:
        iterable (iterable)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within ``num_replicas``.

    Example:
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=0))
        [0, 2, 4, 6, 8]
        >>> list(DistributedSampler(range(10), num_replicas=2, rank=1))
        [1, 3, 5, 7, 9]
    """

    def __init__(self, iterable, num_replicas=None, rank=None):
        self.iterable = iterable
        self.num_replicas = num_replicas
        self.rank = rank

        if num_replicas is None or rank is None:  # pragma: no cover
            if not torch.distributed.is_initialized():
                raise RuntimeError('Requires `torch.distributed` to be initialized.')

            self.num_replicas = (
                torch.distributed.get_world_size() if num_replicas is None else num_replicas)
            self.rank = torch.distributed.get_rank() if rank is None else rank

        if self.rank >= self.num_replicas:
            raise IndexError('`rank` must be smaller than the `num_replicas`.')

    def __iter__(self):
        return iter(
            [e for i, e in enumerate(self.iterable) if (i - self.rank) % self.num_replicas == 0])

    def __len__(self):
        return len(self.iterable)
