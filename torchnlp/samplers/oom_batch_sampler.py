import heapq

from torch.utils.data.sampler import BatchSampler

from torchnlp.utils import get_tensors


def get_number_of_elements(object_):
    """ Get the sum of the number of elements in all tensors stored in `object_`.

    This is particularly useful for sampling the largest objects based on tensor size like in:
    `OomBatchSampler.__init__.get_item_size`.

    Args:
        object (any)

    Returns:
        (int): The number of elements in the `object_`.
    """
    return sum([t.numel() for t in get_tensors(object_)])


class OomBatchSampler(BatchSampler):
    """ Out-of-memory (OOM) batch sampler wraps `batch_sampler` to sample the `num_batches` largest
    batches first in attempt to cause any potential OOM errors early.

    Credits:
    https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        get_item_size (callable): Measure the size of an item given it's index `int`.
        num_batches (int, optional): The number of the large batches to move to the beginning of the
            iteration.
    """

    def __init__(self, batch_sampler, get_item_size, num_batches=5):
        self.batch_sampler = batch_sampler
        self.get_item_size = get_item_size
        self.num_batches = num_batches

    def __iter__(self):
        batches = list(iter(self.batch_sampler))
        largest_batches = heapq.nlargest(
            self.num_batches,
            range(len(batches)),
            key=lambda i: sum([self.get_item_size(j) for j in batches[i]]))
        move_to_front = [batches[i] for i in largest_batches]
        [batches.pop(i) for i in sorted(largest_batches, reverse=True)]
        batches[0:0] = move_to_front
        return iter(batches)

    def __len__(self):
        return len(self.batch_sampler)
