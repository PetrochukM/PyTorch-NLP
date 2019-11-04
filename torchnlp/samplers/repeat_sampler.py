from torch.utils.data.sampler import Sampler


class RepeatSampler(Sampler):
    """ Sampler that repeats forever.

    Background:
        The repeat sampler can be used with the ``DataLoader`` with option to re-use worker
        processes. Learn more here: https://github.com/pytorch/pytorch/issues/15849

    Args:
        sampler (torch.data.utils.sampler.Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
