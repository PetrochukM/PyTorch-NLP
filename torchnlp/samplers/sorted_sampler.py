from torch.utils.data.sampler import Sampler


class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (datasets.Dataset): dataset to sample from
        sort_key (callable): specifies a function of one argument that is used to extract a
          comparison key from each list element
    """

    def __init__(self, data_source, sort_key):
        super().__init__(data_source)
        self.data_source = data_source
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data_source)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data_source)
