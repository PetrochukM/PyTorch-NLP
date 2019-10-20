import random
import pickle

from torchnlp.random import fork_rng
from torchnlp.random import set_seed
from torchnlp.samplers import BalancedSampler
from torchnlp.samplers import DeterministicSampler


def test_deterministic_sampler__nondeterministic_iter():
    with fork_rng(seed=123):
        data = [random.randint(1, 100) for i in range(100)]

    sampler = DeterministicSampler(BalancedSampler(data), random_seed=123)
    assert len(sampler) == len(data)
    samples = [data[i] for i in sampler]
    assert samples[:10] == [3, 35, 99, 43, 67, 82, 66, 68, 100, 14]

    # NOTE: Each iteration is new sample from `sampler`; however, the entire sequence of iterations
    # is deterministic based on the `random_seed=123`
    new_samples = [data[i] for i in sampler]
    assert samples != new_samples


def test_deterministic_sampler__nondeterministic_next():

    class _Sampler():

        def __iter__(self):
            for _ in range(100):
                yield random.randint(1, 100)

    sampler = DeterministicSampler(_Sampler(), random_seed=123)
    assert list(sampler)[:10] == [7, 35, 12, 99, 53, 35, 14, 5, 49, 69]


def test_deterministic_sampler__side_effects():
    """ Ensure that the sampler does not affect random generation after it's finished. """
    set_seed(123)
    pre_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    sampler = DeterministicSampler(list(range(10)), random_seed=123)
    list(iter(sampler))

    post_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    set_seed(123)
    assert pre_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]
    assert post_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]


def test_pickleable():
    data_source = [1, 2, 3, 4, 5]
    sampler = DeterministicSampler(data_source, random_seed=123)
    pickle.dumps(sampler)
