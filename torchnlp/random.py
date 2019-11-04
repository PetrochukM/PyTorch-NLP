from collections import namedtuple
from contextlib import contextmanager

import functools
import random

import numpy as np
import torch

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def get_random_generator_state(cuda=torch.cuda.is_available()):
    """ Get the `torch`, `numpy` and `random` random generator state.

    Args:
        cuda (bool, optional): If `True` saves the `cuda` seed also. Note that getting and setting
            the random generator state for CUDA can be quite slow if you have a lot of GPUs.

    Returns:
        RandomGeneratorState
    """
    return RandomGeneratorState(random.getstate(), torch.random.get_rng_state(),
                                np.random.get_state(),
                                torch.cuda.get_rng_state_all() if cuda else None)


def set_random_generator_state(state):
    """ Set the `torch`, `numpy` and `random` random generator state.

    Args:
        state (RandomGeneratorState)
    """
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)
    if state.torch_cuda is not None and torch.cuda.is_available() and len(
            state.torch_cuda) == torch.cuda.device_count():  # pragma: no cover
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def fork_rng(seed=None, cuda=torch.cuda.is_available()):
    """ Forks the `torch`, `numpy` and `random` random generators, so that when you return, the
    random generators are reset to the state that they were previously in.

    Args:
        seed (int or None, optional): If defined this sets the seed values for the random
            generator fork. This is a convenience parameter.
        cuda (bool, optional): If `True` saves the `cuda` seed also. Getting and setting the random
            generator state can be quite slow if you have a lot of GPUs.
    """
    state = get_random_generator_state(cuda)
    if seed is not None:
        set_seed(seed, cuda)
    try:
        yield
    finally:
        set_random_generator_state(state)


def fork_rng_wrap(function=None, **kwargs):
    """ Decorator alias for `fork_rng`.
    """
    if not function:
        return functools.partial(fork_rng_wrap, **kwargs)

    @functools.wraps(function)
    def wrapper():
        with fork_rng(**kwargs):
            return function()

    return wrapper


def set_seed(seed, cuda=torch.cuda.is_available()):
    """ Set seed values for random generators.

    Args:
        seed (int): Value used as a seed.
        cuda (bool, optional): If `True` sets the `cuda` seed also.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:  # pragma: no cover
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
