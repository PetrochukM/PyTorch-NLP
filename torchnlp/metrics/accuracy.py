import torch

from torchnlp.utils import torch_equals_ignore_index

is_scalar = lambda t: torch.is_tensor(t) and len(t.size()) == 0


def get_accuracy(targets, outputs, k=1, ignore_index=None):
    """ Get the accuracy top-k accuracy between two tensors.

    Args:
      targets (1 - 2D :class:`torch.Tensor`): Target or true vector against which to measure
          saccuracy
      outputs (1 - 3D :class:`torch.Tensor`): Prediction or output vector
      ignore_index (int, optional): Specifies a target index that is ignored

    Returns:
      :class:`tuple` consisting of accuracy (:class:`float`), number correct (:class:`int`) and
      total (:class:`int`)

    Example:

        >>> import torch
        >>> from torchnlp.metrics import get_accuracy
        >>> targets = torch.LongTensor([1, 2, 3, 4, 5])
        >>> outputs = torch.LongTensor([1, 2, 2, 3, 5])
        >>> accuracy, n_correct, n_total = get_accuracy(targets, outputs, ignore_index=3)
        >>> accuracy
        0.8
        >>> n_correct
        4
        >>> n_total
        5
    """
    n_correct = 0.0
    for target, output in zip(targets, outputs):
        if not torch.is_tensor(target) or is_scalar(target):
            target = torch.LongTensor([target])

        if not torch.is_tensor(output) or is_scalar(output):
            output = torch.LongTensor([[output]])

        predictions = output.topk(k=min(k, len(output)), dim=0)[0]
        for prediction in predictions:
            if torch_equals_ignore_index(target, prediction, ignore_index=ignore_index):
                n_correct += 1
                break

    return n_correct / len(targets), n_correct, len(targets)


def get_token_accuracy(targets, outputs, ignore_index=None):
    """ Get the accuracy token accuracy between two tensors.

    Args:
      targets (1 - 2D :class:`torch.Tensor`): Target or true vector against which to measure
          saccuracy
      outputs (1 - 3D :class:`torch.Tensor`): Prediction or output vector
      ignore_index (int, optional): Specifies a target index that is ignored

    Returns:
      :class:`tuple` consisting of accuracy (:class:`float`), number correct (:class:`int`) and
      total (:class:`int`)

    Example:

        >>> import torch
        >>> from torchnlp.metrics import get_token_accuracy
        >>> targets = torch.LongTensor([[1, 1], [2, 2], [3, 3]])
        >>> outputs = torch.LongTensor([[1, 1], [2, 3], [4, 4]])
        >>> accuracy, n_correct, n_total = get_token_accuracy(targets, outputs, ignore_index=3)
        >>> accuracy
        0.75
        >>> n_correct
        3
        >>> n_total
        4
     """
    n_correct = 0.0
    n_total = 0.0
    for target, output in zip(targets, outputs):
        if not torch.is_tensor(target) or is_scalar(target):
            target = torch.LongTensor([target])

        if not torch.is_tensor(output) or is_scalar(output):
            output = torch.LongTensor([[output]])

        if len(target.size()) != len(output.size()):
            prediction = output.max(dim=0)[0].view(-1)
        else:
            prediction = output

        if ignore_index is not None:
            mask = target.ne(ignore_index)
            n_correct += prediction.eq(target).masked_select(mask).sum()
            n_total += mask.sum()
        else:
            n_total += len(target)
            n_correct += prediction.eq(target).sum()

    return n_correct / n_total, n_correct, n_total
