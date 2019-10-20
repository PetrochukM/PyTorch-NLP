from collections import namedtuple
from functools import partial
from unittest import mock

import pickle

import torch

from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors
from torchnlp.utils import flatten_parameters
from torchnlp.utils import get_tensors
from torchnlp.utils import lengths_to_mask
from torchnlp.utils import split_list
from torchnlp.utils import tensors_to
from torchnlp.utils import torch_equals_ignore_index
from torchnlp.utils import get_total_parameters


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
    assert len(tensors) == 5


def test_flatten_parameters():
    rnn = torch.nn.LSTM(10, 20, 2)
    rnn_pickle = pickle.dumps(rnn)
    rnn2 = pickle.loads(rnn_pickle)
    # Check that ``flatten_parameters`` works with a RNN module.
    flatten_parameters(rnn2)


def test_get_total_parameters():
    rnn = torch.nn.LSTM(10, 20, 2)
    assert get_total_parameters(rnn) == 5920


def test_split_list():
    assert split_list([1, 2, 3, 4, 5], (0.6, 0.4)) == [[1, 2, 3], [4, 5]]


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
    mock_tensor.to.called == 1
    mock_tensor.to.reset_mock()

    returned = tensors_to({'t': [mock_tensor]}, device=torch.device('cpu'))
    mock_tensor.to.called == 1
    mock_tensor.to.reset_mock()
    assert isinstance(returned, dict)

    returned = tensors_to([mock_tensor], device=torch.device('cpu'))
    mock_tensor.to.called == 1
    mock_tensor.to.reset_mock()
    assert isinstance(returned, list)

    returned = tensors_to(tuple([mock_tensor]), device=torch.device('cpu'))
    mock_tensor.to.called == 1
    mock_tensor.to.reset_mock()
    assert isinstance(returned, tuple)

    returned = tensors_to(TestTuple(t=mock_tensor), device=torch.device('cpu'))
    mock_tensor.to.called == 1
    mock_tensor.to.reset_mock()
    assert isinstance(returned, TestTuple)


def test_lengths_to_mask():
    assert lengths_to_mask([3]).sum() == 3
    assert lengths_to_mask(torch.tensor(3)).sum() == 3
    assert lengths_to_mask([1, 2, 3]).sum() == 6
    assert lengths_to_mask([1, 2, 3])[0].sum() == 1
    assert lengths_to_mask([1, 2, 3])[0][0].item() == 1
    assert lengths_to_mask(torch.tensor([1, 2, 3]))[0][0].item() == 1
    assert lengths_to_mask(torch.tensor([1.0, 2.0, 3.0]))[0][0].item() == 1
