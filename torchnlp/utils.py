import logging
import inspect
import collections

import random
import torch

from torchnlp.text_encoders import PADDING_INDEX

logger = logging.getLogger(__name__)


def get_tensors(object_):
    """ Get all tensors associated with ``object_``

    Args:
        object_ (any): Any object to look for tensors.

    Returns:
        (list of torch.tensor): List of tensors that are associated with ``object_``.
    """
    if torch.is_tensor(object_):
        return [object_]
    elif isinstance(object_, (str, float, int)):
        return []

    tensors = set()

    if isinstance(object_, collections.Mapping):
        for value in object_.values():
            tensors.update(get_tensors(value))
    elif isinstance(object_, collections.Iterable):
        for value in object_:
            tensors.update(get_tensors(value))
    else:
        members = [
            value for key, value in inspect.getmembers(object_)
            if not isinstance(value, (collections.Callable, type(None)))
        ]
        tensors.update(get_tensors(members))

    return tensors


def sampler_to_iterator(dataset, sampler):
    """ Given a batch sampler or sampler returns examples instead of indices

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        sampler (torch.utils.data.sampler.Sampler): Sampler over the dataset.

    Returns:
        generator over dataset examples
    """
    for sample in sampler:
        if isinstance(sample, (list, tuple)):
            # yield a batch
            yield [dataset[i] for i in sample]
        else:
            # yield a single example
            yield dataset[sample]


def datasets_iterator(*datasets):
    """
    Args:
        *datasets (:class:`list` of :class:`torch.utils.data.Dataset`)

    Returns:
        generator over rows in ``*datasets``
    """
    for dataset in datasets:
        for row in dataset:
            yield row


def pad_tensor(tensor, length, padding_index=PADDING_INDEX):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.
    Args:
        tensor (torch.Tensor [n, *]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.
    Returns
        (torch.Tensor [length, *]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    assert n_padding >= 0
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)


def pad_batch(batch, padding_index=PADDING_INDEX):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.
    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.
    Returns
        torch.Tensor, list of int: Padded tensors and original lengths of tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    padded = torch.stack(padded, dim=0).contiguous()
    return padded, lengths


def flatten_parameters(model):
    """ ``flatten_parameters`` of a RNN model loaded from disk. """
    model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)


def shuffle(list_, random_seed=123):
    """ Shuffle list deterministically based on ``random_seed``.

    **Reference:**
    https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result

    Example:
        >>> a = [1, 2, 3, 4, 5]
        >>> b = [1, 2, 3, 4, 5]
        >>> shuffle(a, random_seed=456)
        >>> shuffle(b, random_seed=456)
        >>> a == b
        True
        >>> a, b
        ([1, 3, 2, 5, 4], [1, 3, 2, 5, 4])

    Args:
        list_ (list): List to be shuffled.
        random_seed (int): Random seed used to shuffle.
    Returns:
        None:
    """
    random.Random(random_seed).shuffle(list_)


def resplit_datasets(dataset, other_dataset, random_seed=None, split=None):
    """Deterministic shuffle and split algorithm.

    Given the same two datasets and the same ``random_seed``, the split happens the same exact way
    every call.

    Args:
        dataset (lib.datasets.Dataset): First dataset.
        other_dataset (lib.datasets.Dataset): Another dataset.
        random_seed (int, optional): Seed to control the shuffle of both datasets.
        split (float, optional): If defined it is the percentage of rows that first dataset gets
            after split otherwise the original proportions are kept.

    Returns:
        :class:`lib.datasets.Dataset`, :class:`lib.datasets.Dataset`: Resplit datasets.
    """
    # Prevent circular dependency
    from torchnlp.datasets import Dataset

    concat = dataset.rows + other_dataset.rows
    shuffle(concat, random_seed=random_seed)
    if split is None:
        return Dataset(concat[:len(dataset)]), Dataset(concat[len(dataset):])
    else:
        split = max(min(round(len(concat) * split), len(concat)), 0)
        return Dataset(concat[:split]), Dataset(concat[split:])


def torch_equals_ignore_index(tensor, tensor_other, ignore_index=None):
    """
    Compute ``torch.equal`` with the optional mask parameter.

    Args:
        ignore_index (int, optional): Specifies a ``tensor`` index that is ignored.

    Returns:
        (bool) Returns ``True`` if target and prediction are equal.
    """
    if ignore_index is not None:
        assert tensor.size() == tensor_other.size()
        mask_arr = tensor.ne(ignore_index)
        tensor = tensor.masked_select(mask_arr)
        tensor_other = tensor_other.masked_select(mask_arr)

    return torch.equal(tensor, tensor_other)
