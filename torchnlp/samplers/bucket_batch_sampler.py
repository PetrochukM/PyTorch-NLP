import random

from torch.utils.data.sampler import BatchSampler

from torchnlp.samplers.noisy_sorted_sampler import NoisySortedSampler


class BucketBatchSampler(BatchSampler):
    """
    Reference:
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py
    https://github.com/pytorch/text/tree/master/torchtext/data/iterators/#BucketIterator

    `BucketIterator` pools together examples with a similar size length to reduce the padding
    required for each batch. `BucketIterator` typically also includes the ability to add noise to
    the pooling.

    The functionality has been replicated as a `Sampler` to be used with a
    `torch.data.utils.DataLoader`.
    """

    def __init__(self,
                 data_source,
                 sort_key,
                 batch_size,
                 sort_key_noise=0.25,
                 last_batch_first=True,
                 shuffle=True,
                 drop_last=False):
        self.last_batch_first = last_batch_first
        self.shuffle = shuffle
        super().__init__(
            NoisySortedSampler(
                data_source=data_source, sort_key=sort_key, sort_key_noise=sort_key_noise),
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
