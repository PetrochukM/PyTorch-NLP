from functools import partial

import pickle

import torch

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler

from torchnlp.samplers import get_number_of_elements
from torchnlp.samplers import OomBatchSampler


def test_oom_batch_sampler():
    data = list(range(-4, 14))
    sampler = SequentialSampler(data)
    batch_sampler = BatchSampler(sampler, 4, False)
    oom_sampler = OomBatchSampler(batch_sampler, lambda i: data[i], num_batches=3)
    list_ = list(oom_sampler)
    # The largest batches are first
    assert [data[i] for i in list_[0]] == [8, 9, 10, 11]
    assert [data[i] for i in list_[1]] == [12, 13]
    assert [data[i] for i in list_[2]] == [4, 5, 6, 7]
    assert len(list_) == 5


def test_get_number_of_elements():
    assert get_number_of_elements([torch.randn(5, 5), torch.randn(4, 4)]) == 41


def get_index(data, i):
    return data[i]


def test_pickleable():
    data = list(range(-4, 14))
    sampler = SequentialSampler(data)
    batch_sampler = BatchSampler(sampler, 4, False)
    get_data = partial(get_index, data)
    oom_sampler = OomBatchSampler(batch_sampler, get_data, num_batches=3)
    pickle.dumps(oom_sampler)
