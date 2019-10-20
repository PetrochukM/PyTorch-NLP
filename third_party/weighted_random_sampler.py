import torch

from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes


class WeightedRandomSampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        # NOTE: Adapted `WeightedRandomSampler` to accept `num_samples=0`.
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples < 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        if self.num_samples == 0:
            return iter([])

        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
