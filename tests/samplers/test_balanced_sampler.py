from collections import Counter

import pickle
import pytest
import random

from torchnlp.random import fork_rng_wrap
from torchnlp.samplers import BalancedSampler


# NOTE: `fork_rng_wrap` to ensure the tests never randomly fail due to an rare sampling.
@fork_rng_wrap(seed=123)
def test_balanced_sampler():
    data = ['a', 'a', 'b', 'b', 'b', 'c']
    num_samples = 10000
    sampler = BalancedSampler(data, replacement=True, num_samples=num_samples)
    assert len(sampler) == num_samples
    samples = [data[i] for i in sampler]
    assert len(samples) == num_samples
    counts = Counter(samples)
    assert counts['a'] / num_samples == pytest.approx(.33, 0.2)
    assert counts['b'] / num_samples == pytest.approx(.33, 0.2)
    assert counts['c'] / num_samples == pytest.approx(.33, 0.2)


@fork_rng_wrap(seed=123)
def test_balanced_sampler__weighted():
    data = [('a', 0), ('a', 1), ('a', 2), ('b', 2), ('c', 1)]
    num_samples = 10000
    sampler = BalancedSampler(
        data,
        replacement=True,
        num_samples=num_samples,
        get_weight=lambda e: e[1],
        get_class=lambda e: e[0])
    samples = [data[i] for i in sampler]
    counts = Counter(samples)
    assert counts[('a', 2)] / num_samples == pytest.approx(.22, 0.2)
    assert counts[('a', 1)] / num_samples == pytest.approx(.11, 0.2)
    assert counts[('a', 0)] / num_samples == 0.0
    assert counts[('b', 2)] / num_samples == pytest.approx(.33, 0.2)
    assert counts[('c', 1)] / num_samples == pytest.approx(.33, 0.2)


@fork_rng_wrap(seed=123)
def test_balanced_sampler__nondeterministic():
    data = [random.randint(1, 100) for i in range(100)]
    sampler = BalancedSampler(data)
    samples = [data[i] for i in sampler]
    new_samples = [data[i] for i in sampler]
    assert new_samples != samples


def test_pickleable():
    data_source = [1, 2, 3, 4, 5]
    sampler = BalancedSampler(data_source)
    pickle.dumps(sampler)
