from collections import namedtuple
from functools import partial
from unittest import mock

import pickle

import torch

from torchnlp.datasets import Dataset
from torchnlp.utils import collate_tensors
from torchnlp.utils import flatten_parameters
from torchnlp.utils import get_tensors
from torchnlp.utils import resplit_datasets
from torchnlp.utils import shuffle
from torchnlp.utils import tensors_to
from torchnlp.utils import torch_equals_ignore_index
from torchnlp.encoders.text import stack_and_pad_tensors


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


TestTuple = namedtuple('TestTuple', ['t'])


def test_collate_tensors():

    tensor = torch.Tensor(1)
    collate_sequences = partial(collate_tensors, stack_tensors=stack_and_pad_tensors)
    assert collate_sequences([tensor, tensor])[0].shape == (2, 1)
    assert collate_sequences([[tensor], [tensor]])[0][0].shape == (2, 1)
    assert collate_sequences([{'t': tensor}, {'t': tensor}])['t'][0].shape == (2, 1)
    assert collate_sequences([TestTuple(t=tensor), TestTuple(t=tensor)]).t[0].shape == (2, 1)
    assert collate_sequences(['test', 'test']) == ['test', 'test']


@mock.patch('torch.is_tensor')
def test_tensors_to(mock_is_tensor):
    TestTuple = namedtuple('TestTuple', ['t'])

    mock_tensor = mock.Mock()
    mock_is_tensor.side_effect = lambda m, **kwargs: m == mock_tensor
    tensors_to(mock_tensor, device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()

    returned = tensors_to({'t': [mock_tensor]}, device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, dict)

    returned = tensors_to([mock_tensor], device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, list)

    returned = tensors_to(tuple([mock_tensor]), device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, tuple)

    returned = tensors_to(TestTuple(t=mock_tensor), device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, TestTuple)
