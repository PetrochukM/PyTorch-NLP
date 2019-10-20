import random

from torch.utils.data.sampler import Sampler

from torchnlp.utils import identity


def _uniform_noise(_):
    return random.uniform(-1, 1)


class NoisySortedSampler(Sampler):
    """ Samples elements sequentially with noise.

    **Background**

        ``NoisySortedSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        `AllenNLP` and `torchtext`. A ``BucketIterator`` pools together examples with a similar size
        length to reduce the padding required for each batch. ``BucketIterator`` also includes the
        ability to add noise to the pooling.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        data (iterable): Data to sample from.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
        get_noise (callable): Noise added to each numerical ``sort_key``.

    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123)
        >>>
        >>> import random
        >>> get_noise = lambda i: round(random.uniform(-1, 1))
        >>> list(NoisySortedSampler(range(10), sort_key=lambda i: i, get_noise=get_noise))
        [0, 1, 2, 3, 5, 4, 6, 7, 9, 8]
    """

    def __init__(self, data, sort_key=identity, get_noise=_uniform_noise):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        self.get_noise = get_noise

    def __iter__(self):
        zip_ = []
        for i, row in enumerate(self.data):
            value = self.get_noise(row) + self.sort_key(row)
            zip_.append(tuple([i, value]))
        zip_ = sorted(zip_, key=lambda r: r[1])
        return iter([item[0] for item in zip_])

    def __len__(self):
        return len(self.data)
