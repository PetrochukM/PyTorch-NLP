import pickle

import torch

from torchnlp.datasets import Dataset
from torchnlp.utils import flatten_parameters
from torchnlp.utils import resplit_datasets
from torchnlp.utils import shuffle
from torchnlp.utils import torch_equals_ignore_index
from torchnlp.utils import get_tensors


class GetTensorsObjectMock(object):

    class_attribute = torch.tensor([4, 5])

    def __init__(self, recurse=True):
        self.noise_int = 3
        self.noise_str = 'abc'
        self.instance_attribute = frozenset([torch.tensor([6, 7])])
        if recurse:
            self.object_ = GetTensorsObjectMock(recurse=False)

    @property
    def property_(self):
        return torch.tensor([7, 8])


def test_get_tensors_list():
    list_ = [torch.tensor([1, 2]), torch.tensor([2, 3])]
    tensors = get_tensors(list_)
    assert len(tensors) == 2


def test_get_tensors_dict():
    list_ = [{'t': torch.tensor([1, 2])}, torch.tensor([2, 3])]
    tensors = get_tensors(list_)
    assert len(tensors) == 2


def test_get_tensors_tuple():
    tuple_ = tuple([{'t': torch.tensor([1, 2])}, torch.tensor([2, 3])])
    tensors = get_tensors(tuple_)
    assert len(tensors) == 2


def test_get_tensors_object():
    object_ = GetTensorsObjectMock()
    tensors = get_tensors(object_)
    assert len(tensors) == 6


def test_shuffle():
    a = [1, 2, 3, 4, 5]
    # Always shuffles the same way
    shuffle(a)
    assert a == [4, 2, 5, 3, 1]


def test_flatten_parameters():
    rnn = torch.nn.LSTM(10, 20, 2)
    rnn_pickle = pickle.dumps(rnn)
    rnn2 = pickle.loads(rnn_pickle)
    # Check that ``flatten_parameters`` works with a RNN module.
    flatten_parameters(rnn2)


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
