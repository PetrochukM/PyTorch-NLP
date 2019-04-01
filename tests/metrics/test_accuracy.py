import torch

from torchnlp.metrics import get_accuracy
from torchnlp.metrics import get_token_accuracy


def test_get_accuracy():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([1, 2, 3, 3])
    accuracy, _, _ = get_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_token_accuracy():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([1, 2, 3, 3])
    accuracy, _, _ = get_token_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_accuracy_2d_2d():
    targets = torch.LongTensor([[1], [2], [3], [4]])
    outputs = torch.LongTensor([[1], [2], [3], [3]])
    accuracy, _, _ = get_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_token_accuracy_2d_2d():
    targets = torch.LongTensor([[1], [2], [3], [4]])
    outputs = torch.LongTensor([[1], [2], [3], [3]])
    accuracy, _, _ = get_token_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_token_accuracy_2d_2d_2d_2d():
    targets = torch.LongTensor([[1, 1], [2, 2], [3, 3]])
    outputs = torch.LongTensor([[1, 1], [2, 3], [4, 4]])
    accuracy, _, _ = get_token_accuracy(targets, outputs, ignore_index=3)
    assert accuracy == 0.75


def test_get_accuracy_2d_3d():
    targets = torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    outputs = torch.LongTensor([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]],
                                [[3, 3], [3, 3]]])
    accuracy, _, _ = get_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_token_accuracy_2d_3d():
    targets = torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    outputs = torch.LongTensor([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]],
                                [[3, 3], [3, 3]]])
    accuracy, _, _ = get_token_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_accuracy_2d_3d_top_k():
    targets = torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    outputs = torch.LongTensor([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]],
                                [[3, 3], [4, 4]]])
    accuracy, _, _ = get_accuracy(targets, outputs, k=3)
    assert accuracy == 1.0


def test_get_accuracy_1d_2d():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([[1], [2], [3], [3]])
    accuracy, _, _ = get_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_token_accuracy_1d_2d():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([[1], [2], [3], [3]])
    accuracy, _, _ = get_token_accuracy(targets, outputs)
    assert accuracy == 0.75


def test_get_accuracy_1d_2d_top_k():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([[1, 1], [2, 2], [3, 3], [3, 4]])
    accuracy, _, _ = get_accuracy(targets, outputs, k=3)
    assert accuracy == 1.0


def test_get_accuracy_ignore_index():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([1, 2, 3, 3])
    accuracy, _, _ = get_accuracy(targets, outputs, ignore_index=4)
    assert accuracy == 1.0


def test_get_token_accuracy_ignore_index():
    targets = torch.LongTensor([1, 2, 3, 4])
    outputs = torch.LongTensor([1, 2, 3, 3])
    accuracy, _, _ = get_token_accuracy(targets, outputs, ignore_index=4)
    assert accuracy == 1.0
