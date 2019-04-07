import logging
import inspect
import collections

import random
import torch

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

    if isinstance(object_, collections.abc.Mapping):
        for value in object_.values():
            tensors.update(get_tensors(value))
    elif isinstance(object_, collections.abc.Iterable):
        for value in object_:
            tensors.update(get_tensors(value))
    else:
        members = [
            value for key, value in inspect.getmembers(object_)
            if not isinstance(value, (collections.abc.Callable, type(None)))
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


def is_namedtuple(object_):
    return hasattr(object_, '_asdict') and isinstance(object_, tuple)


def lengths_to_mask(*lengths, **kwargs):
    """ Given a list of lengths, create a batch mask.

    Example:
        >>> lengths_to_mask([1, 2, 3])
        tensor([[1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]], dtype=torch.uint8)
        >>> lengths_to_mask([1, 2, 2], [1, 2, 2])
        tensor([[[1, 0],
                 [0, 0]],
        <BLANKLINE>
                [[1, 1],
                 [1, 1]],
        <BLANKLINE>
                [[1, 1],
                 [1, 1]]], dtype=torch.uint8)

    Args:
        *lengths (list of int or torch.Tensor)
        **kwargs: Keyword arguments passed to ``torch.zeros`` upon initially creating the returned
          tensor.

    Returns:
        torch.ByteTensor
    """
    # Squeeze to deal with random additional dimensions
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]

    # For cases where length is a scalar, this needs to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][[slice(int(l)) for l in length]].fill_(1)
    return mask.byte()


def collate_tensors(batch, stack_tensors=torch.stack):
    """ Collate a list of type ``k`` (dict, namedtuple, list, etc.) with tensors.

    Inspired by:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L31

    Args:
        batch (list of k): List of rows of type ``k``.
        stack_tensors (callable): Function to stack tensors into a batch.

    Returns:
        k: Collated batch of type ``k``.

    Example use case:
        This is useful with ``torch.utils.data.dataloader.DataLoader`` which requires a collate
        function. Typically, when collating sequences you'd set
        ``collate_fn=partial(collate_tensors, stack_tensors=encoders.text.stack_and_pad_tensors)``.

    Example:

        >>> import torch
        >>> batch = [
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ... ]
        >>> collated = collate_tensors(batch)
        >>> {k: t.size() for (k, t) in collated.items()}
        {'column_a': torch.Size([2, 5]), 'column_b': torch.Size([2, 5])}
    """
    if all([torch.is_tensor(b) for b in batch]):
        return stack_tensors(batch)
    if (all([isinstance(b, dict) for b in batch]) and
            all([b.keys() == batch[0].keys() for b in batch])):
        return {key: collate_tensors([d[key] for d in batch], stack_tensors) for key in batch[0]}
    elif all([is_namedtuple(b) for b in batch]):  # Handle ``namedtuple``
        return batch[0].__class__(**collate_tensors([b._asdict() for b in batch], stack_tensors))
    elif all([isinstance(b, list) for b in batch]):
        # Handle list of lists such each list has some column to be batched, similar to:
        # [['a', 'b'], ['a', 'b']] → [['a', 'a'], ['b', 'b']]
        transposed = zip(*batch)
        return [collate_tensors(samples, stack_tensors) for samples in transposed]
    else:
        return batch


def tensors_to(tensors, *args, **kwargs):
    """ Apply ``torch.Tensor.to`` to tensors in a generic data structure.

    Inspired by:
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L31

    Args:
        tensors (tensor, dict, list, namedtuple or tuple): Data structure with tensor values to
            move.
        *args: Arguments passed to ``torch.Tensor.to``.
        **kwargs: Keyword arguments passed to ``torch.Tensor.to``.

    Example use case:
        This is useful as a complementary function to ``collate_tensors``. Following collating,
        it's important to move your tensors to the appropriate device.

    Returns:
        The inputted ``tensors`` with ``torch.Tensor.to`` applied.

    Example:

        >>> import torch
        >>> batch = [
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ...   { 'column_a': torch.randn(5), 'column_b': torch.randn(5) },
        ... ]
        >>> tensors_to(batch, torch.device('cpu'))  # doctest: +ELLIPSIS
        [{'column_a': tensor(...}]
    """
    if torch.is_tensor(tensors):
        return tensors.to(*args, **kwargs)
    elif isinstance(tensors, dict):
        return {k: tensors_to(v, *args, **kwargs) for k, v in tensors.items()}
    elif hasattr(tensors, '_asdict') and isinstance(tensors, tuple):  # Handle ``namedtuple``
        return tensors.__class__(**tensors_to(tensors._asdict(), *args, **kwargs))
    elif isinstance(tensors, list):
        return [tensors_to(t, *args, **kwargs) for t in tensors]
    elif isinstance(tensors, tuple):
        return tuple([tensors_to(t, *args, **kwargs) for t in tensors])
    else:
        return tensors
