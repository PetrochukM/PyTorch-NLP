import random

from torch.utils.data.sampler import Sampler


class NoisySortedSampler(Sampler):
    """Samples elements sequentially with noise.

    Reference and inspiration:
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

    Arguments:
        data_source (datasets.Dataset): dataset to sample from
        sort_key (callable): specifies a function of one argument that is used to extract a
          comparison key from each list element
    """

    def __init__(self, data_source, sort_key, sort_key_noise=0.25):
        super().__init__(data_source)
        self.data_source = data_source
        self.sort_key = sort_key
        zip_ = []
        for i, row in enumerate(self.data_source):
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
        return len(self.data_source)
