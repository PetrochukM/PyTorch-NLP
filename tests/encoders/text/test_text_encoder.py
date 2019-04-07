import pytest
import torch

from torchnlp.encoders.text import DEFAULT_PADDING_INDEX
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text import pad_tensor


def test_pad_tensor():
    padded = pad_tensor(torch.LongTensor([1, 2, 3]), 5, DEFAULT_PADDING_INDEX)
    assert padded.tolist() == [1, 2, 3, DEFAULT_PADDING_INDEX, DEFAULT_PADDING_INDEX]


def test_pad_tensor_multiple_dim():
    padded = pad_tensor(torch.LongTensor(1, 2, 3), 5, DEFAULT_PADDING_INDEX)
    assert padded.size() == (5, 2, 3)
    assert padded[1].sum().item() == pytest.approx(0)


def test_pad_tensor_multiple_dim_float_tensor():
    padded = pad_tensor(torch.FloatTensor(778, 80), 804, DEFAULT_PADDING_INDEX)
    assert padded.size() == (804, 80)
    assert padded[-1].sum().item() == pytest.approx(0)
    assert padded.type() == 'torch.FloatTensor'


def test_stack_and_pad_tensors():
    batch = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2]), torch.LongTensor([1])]
    padded, lengths = stack_and_pad_tensors(batch, DEFAULT_PADDING_INDEX)
    padded = [r.tolist() for r in padded]
    assert padded == [[1, 2, 3], [1, 2, DEFAULT_PADDING_INDEX],
                      [1, DEFAULT_PADDING_INDEX, DEFAULT_PADDING_INDEX]]
    assert lengths.tolist() == [3, 2, 1]


def test_stack_and_pad_tensors__dim():
    batch_size = 3
    batch = [torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2])]
    padded, lengths = stack_and_pad_tensors(batch, DEFAULT_PADDING_INDEX, dim=1)
    assert padded.shape == (4, batch_size)
    assert lengths.shape == (1, batch_size)
    assert lengths.tolist() == [[4, 3, 2]]
    assert padded.tolist() == [[1, 1, 1], [2, 2, 2], [3, 3, DEFAULT_PADDING_INDEX],
                               [4, DEFAULT_PADDING_INDEX, DEFAULT_PADDING_INDEX]]
