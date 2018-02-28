import torch

from torchnlp.utils import torch_equals_ignore_index


def get_accuracy(targets, outputs, k=1, ignore_index=None):
    """
    Compute the accuracy of o == t {o \in outputs and t \in targets.

    Args:
      targets (list of tensors)
      outputs (list of tensors)
      ignore_index (optional, int): Specifies a target value that is ignored computing equality
    Returns:
      accuracy (float)
      number correct (int)
      total (int)
    """
    n_correct = 0.0
    for target, output in zip(targets, outputs):
        if not torch.is_tensor(target):
            target = torch.LongTensor([target])

        if not torch.is_tensor(output):
            output = torch.LongTensor([[output]])

        predictions = output.topk(k=min(k, len(output)), dim=0)[0]
        for prediction in predictions:
            if not torch.is_tensor(prediction):
                prediction = torch.LongTensor([prediction])

            if torch_equals_ignore_index(target, prediction, ignore_index=ignore_index):
                n_correct += 1
                break

    return n_correct / len(targets), n_correct, len(targets)


def get_token_accuracy(targets, outputs, ignore_index=None):
    """ Compute the token accuracy.

    Args:
      targets (list of tensors)
      outputs (list of tensors)
      ignore_index (optional, int): Specifies a target value that is ignored computing equality
    Returns:
      accuracy (float)
      number correct (int)
      total (int)
     """
    n_correct = 0.0
    n_total = 0.0
    for target, output in zip(targets, outputs):
        if not torch.is_tensor(target):
            target = torch.LongTensor([target])

        if not torch.is_tensor(output):
            output = torch.LongTensor([[output]])

        prediction = output.max(dim=0)[0].view(-1)
        if ignore_index is not None:
            mask = target.ne(ignore_index)
            n_correct += prediction.eq(target).masked_select(mask).sum()
            n_total += mask.sum()
        else:
            n_total += len(target)
            n_correct += prediction.eq(target).sum()

    return n_correct / n_total, n_correct, n_total
