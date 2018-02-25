from torch.utils.data.sampler import Sampler


class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
        sort_key (callable): callable that returns from one row of the data_source a sortable
            value
    """

    def __init__(self, data_source, sort_key, sort_noise=0.1):
        self.data_source = data_source
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data_source)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data_source)
