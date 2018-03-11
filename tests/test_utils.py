import torch

from torchnlp.datasets import Dataset
from torchnlp.utils import resplit_datasets
from torchnlp.utils import torch_equals_ignore_index
from torchnlp.utils import shuffle


def test_shuffle():
    a = Dataset([{'r': 1}, {'r': 2}, {'r': 3}, {'r': 4}, {'r': 5}])
    # Always shuffles the same way
    shuffle(a)
    assert a == Dataset([{'r': 4}, {'r': 2}, {'r': 5}, {'r': 3}, {'r': 1}])


def test_resplit_datasets():
    a = Dataset([{'r': 1}, {'r': 2}, {'r': 3}, {'r': 4}, {'r': 5}])
    b = Dataset([{'r': 6}, {'r': 7}, {'r': 8}, {'r': 9}, {'r': 10}])
    # Test determinism
    a, b = resplit_datasets(a, b, random_seed=123)
    assert list(a) == [{'r': 9}, {'r': 8}, {'r': 6}, {'r': 10}, {'r': 3}]
    assert list(b) == [{'r': 4}, {'r': 7}, {'r': 2}, {'r': 5}, {'r': 1}]


def test_resplit_datasets_cut():
    a = Dataset([{'r': 1}, {'r': 2}, {'r': 3}, {'r': 4}, {'r': 5}])
    b = Dataset([{'r': 6}, {'r': 7}, {'r': 8}, {'r': 9}, {'r': 10}])
    a, b = resplit_datasets(a, b, random_seed=123, split=0.3)
    assert len(a) == 3
    assert len(b) == 7


def test_torch_equals_ignore_index():
    source = torch.LongTensor([1, 2, 3])
    target = torch.LongTensor([1, 2, 4])
    assert torch_equals_ignore_index(source, target, ignore_index=3)
    assert not torch_equals_ignore_index(source, target)
