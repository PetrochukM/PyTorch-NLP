import heapq
import pickle

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler

from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.samplers.shuffle_batch_sampler import ShuffleBatchSampler


class BucketBatchSampler(object):
    """Samples a noisy sorted mini-batch of indices from a data source.

    In order to introduce, noise into a sorted mini-batch, we use a bucketing technique from
    `torchtext`. First, partition data in buckets of size 100 * `batch_size`. The examples inside
    the buckets are sorted using `sort_key` and batched. Finally, those batches are shuffled.

    Background:
        BucketBatchSampler is similar to a BucketIterator found in popular libraries like `AllenNLP`
        and `torchtext`. A BucketIterator pools together examples with a similar size length to
        reduce the padding required for each batch. BucketIterator also includes the ability to add
        noise to the pooling.

        AllenNLP Implementation:
        https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

        torchtext Implementation:
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        data (iterable): Data to sample from.
        batch_size (int): Size of mini-batch.
        sort_key (callable): specifies a function of one argument that is used to extract a
          comparison key from each list element
        drop_last (bool, optional): If ``True``, the sampler will drop the last batch if its size
            would be less than ``batch_size``
        biggest_batch_first (bool, optional): If ``True``, the sampler will use cPickle to
            approximate the memory footprint of each batch and attempt to return the 5 biggest
            batches first.

            This is largely for testing, to see how large of a batch you can safely use with your
            GPU. This will let you try out the biggest batch that you have in the data `first`, so
            that if you're going to run out of memory, you know it early, instead of waiting
            through the whole epoch to find out at the end that you're going to crash.

            Credits:
            https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43
        bucket_size_multiplier (int): Batch size multiplier to determine the bucket size.

    Example:
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    """

    def __init__(
            self,
            data,
            batch_size,
            sort_key,
            drop_last=False,
            biggest_batches_first=True,
            bucket_size_multiplier=100,
    ):
        self.biggest_batches_first = biggest_batches_first
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data = data

        self.bucket_size_multiplier = bucket_size_multiplier
        self.bucket_sampler = BatchSampler(
            RandomSampler(data), batch_size * bucket_size_multiplier, False)

    def __iter__(self):

        def get_batches():
            """ Get bucketed batches """
            for bucket in self.bucket_sampler:
                for batch in ShuffleBatchSampler(
                        SortedSampler(bucket, lambda i: self.sort_key(self.data[i])),
                        self.batch_size,
                        drop_last=self.drop_last,
                        shuffle=True):
                    batch = [bucket[i] for i in batch]

                    # Should only be triggered once
                    if len(batch) < self.batch_size and self.drop_last:
                        continue

                    yield batch

        if not self.biggest_batches_first:
            return get_batches()
        else:
            batches = list(get_batches())
            indices = heapq.nlargest(
                5,
                range(len(batches)),
                key=lambda i: len(pickle.dumps([self.data[j] for j in batches[i]])))
            front = [batches[i] for i in indices]
            for i in sorted(indices, reverse=True):
                batches.pop(i)
            batches[0:0] = front
            return iter(batches)
