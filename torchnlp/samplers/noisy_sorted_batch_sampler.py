import random

from torch.utils.data.sampler import BatchSampler

from torchnlp.samplers.noisy_sorted_sampler import NoisySortedSampler


class NoisySortedBatchSampler(BatchSampler):
    """ Batch of indices are sampled from a noisy sorting of the data.

    Batches are sampled on top of a noisy sorting. ``sort_key`` returns a number by which elements
    are sorted onto which noise from a uniform distribution up to ``sort_key_noise`` is added.

    **Background**

        NoisySortedBatchSampler is similar to a BucketIterator found in popular libraries like
        `AllenNLP` and `torchtext`. A BucketIterator pools together examples with a similar size
        length to reduce the padding required for each batch. BucketIterator also includes the
        ability to add noise to the pooling.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        data (iterable): Iterable data.
        batch_size (int): Size of mini-batch.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        sort_key_noise (float): Maximum noise added to the numerical `sort_key`.
        last_batch_first (bool, optional): If ``True``, the sampler will append the last batch
            first. Only helpful if the `sort_key` approximates GPU memory.

            This is largely for testing, to see how large of a batch you can safely use with your
            GPU. This will let you try out the biggest batch that you have in the data `first`, so
            that if you're going to run out of memory, you know it early, instead of waiting
            through the whole epoch to find out at the end that you're going to crash.

            Credits:
            https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43
        shuffle (bool, optional): If ``True``, the batches are shuffled.

    Example:
        >>> list(NoisySortedBatchSampler(range(10), batch_size=3, drop_last=False))
        [[9], [0, 1, 2], [3, 4, 5], [6, 8, 7]]
        >>> list(NoisySortedBatchSampler(range(10), batch_size=3, drop_last=True))
        [[7, 9, 6], [0, 1, 2], [3, 4, 5]]

    """

    def __init__(self,
                 data,
                 batch_size,
                 drop_last,
                 sort_key=lambda e: e,
                 sort_key_noise=0.25,
                 last_batch_first=True,
                 shuffle=True):
        self.last_batch_first = last_batch_first
        self.shuffle = shuffle
        super().__init__(
            NoisySortedSampler(data=data, sort_key=sort_key, sort_key_noise=sort_key_noise),
            batch_size, drop_last)

    def __iter__(self):
        batches = list(super().__iter__())
        if self.last_batch_first:
            last_batch = batches.pop()
        if self.shuffle:
            random.shuffle(batches)
        if self.last_batch_first:
            batches.insert(0, last_batch)
        return iter(batches)
