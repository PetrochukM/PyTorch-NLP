import random

import torch
import numpy

from torchnlp.random import fork_rng
from torchnlp.random import fork_rng_wrap
from torchnlp.random import get_random_generator_state
from torchnlp.random import set_random_generator_state
from torchnlp.random import set_seed


def test_random_generator_state():
    # TODO: Test `torch.cuda` random as well.
    state = get_random_generator_state()
    randint = random.randint(1, 2**31)
    numpy_randint = numpy.random.randint(1, 2**31)
    torch_randint = int(torch.randint(1, 2**31, (1,)))

    set_random_generator_state(state)
    post_randint = random.randint(1, 2**31)
    post_numpy_randint = numpy.random.randint(1, 2**31)
    post_torch_randint = int(torch.randint(1, 2**31, (1,)))

    assert randint == post_randint
    assert numpy_randint == post_numpy_randint
    assert torch_randint == post_torch_randint


def test_set_seed__smoke_test():
    set_seed(123)


def test_fork_rng_wrap():
    set_seed(123)
    pre_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    @fork_rng_wrap()
    def func():
        random.randint(1, 2**31)

    func()

    post_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    set_seed(123)
    assert pre_randint != post_randint
    assert pre_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]
    assert post_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]


def test_fork_rng():
    set_seed(123)
    pre_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    with fork_rng(seed=123):
        random.randint(1, 2**31)

    post_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    set_seed(123)
    assert pre_randint != post_randint
    assert pre_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]
    assert post_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]
