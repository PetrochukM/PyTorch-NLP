from contextlib import contextmanager

from torch.utils.data.sampler import Sampler

import torch

from torchnlp.random import fork_rng
from torchnlp.random import get_random_generator_state
from torchnlp.random import set_random_generator_state
from torchnlp.random import set_seed


class DeterministicSampler(Sampler):
    """ Maintains a random state such that `sampler` returns the same output every process.

    Args:
        sampler (torch.data.utils.sampler.Sampler)
        random_seed (int)
        cuda (bool, optional): If `True` this sampler forks the random state of CUDA as well.
    """

    def __init__(self, sampler, random_seed, cuda=torch.cuda.is_available()):
        self.sampler = sampler
        self.rng_state = None
        self.random_seed = random_seed
        self.cuda = cuda

    @contextmanager
    def _fork_rng(self):
        with fork_rng(cuda=self.cuda):
            if self.rng_state is not None:
                set_random_generator_state(self.rng_state)
            else:
                set_seed(self.random_seed)

            try:
                yield
            finally:
                self.rng_state = get_random_generator_state(cuda=self.cuda)

    def __iter__(self):
        with self._fork_rng():
            iterator = iter(self.sampler)

        while True:
            try:
                with self._fork_rng():
                    sample = next(iterator)
                yield sample
            except StopIteration:
                break

    def __len__(self):
        return len(self.sampler)
