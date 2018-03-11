import random

from torch.utils.data.sampler import Sampler


class NoisySortedSampler(Sampler):
    """Samples elements sequentially with noise.

    Reference and inspiration:
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

    Args:
        data (iterable): Data to sample from.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
        sort_key_noise (float): Maximum noise added to the numerical `sort_key`.

    Example:
        >>> list(NoisySortedSampler(range(10), sort_key=lambda i: i, sort_key_noise=0.25))
        [0, 1, 2, 3, 5, 4, 6, 8, 7, 9]
    """

    def __init__(self, data, sort_key, sort_key_noise=0.25):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = []
        for i, row in enumerate(self.data):
            value = self.sort_key(row)
            noise_value = value * sort_key_noise
            noise = random.uniform(-noise_value, noise_value)
            value = noise + value
            zip_.append(tuple([i, value]))
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)
