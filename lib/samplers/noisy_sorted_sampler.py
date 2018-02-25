import random

from lib.samplers.sorted_sampler import SortedSampler


class NoisySortedSampler(SortedSampler):
    """Samples elements sequentially, always in the same order.

    Reference and inspiration:
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

    Arguments:
        data_source (Dataset): dataset to sample from
        sort_key (callable -> int): callable that returns from one row of the data_source a int
    """

    def __init__(self, data_source, sort_key, sort_key_noise=0.1):
        self.data_source = data_source
        self.sort_key = sort_key
        zip = []
        for i, row in enumerate(self.data_source):
            value = self.sort_key(row)
            noise_value = value * sort_key_noise
            noise = random.uniform(-noise_value, noise_value)
            value = noise + value
            zip.append(tuple([i, value]))
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]
