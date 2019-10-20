import logging
import inspect
import collections

import torch

logger = logging.getLogger(__name__)


def _get_tensors(object_, seen=set()):
    if torch.is_tensor(object_):
        return [object_]

    elif isinstance(object_, (str, float, int)) or id(object_) in seen:
        return []

    seen.add(id(object_))
    tensors = set()

    if isinstance(object_, collections.abc.Mapping):
        for value in object_.values():
            tensors.update(_get_tensors(value, seen))
    elif isinstance(object_, collections.abc.Iterable):
        for value in object_:
            tensors.update(_get_tensors(value, seen))
    else:
        members = [
            value for key, value in inspect.getmembers(object_)
            if not isinstance(value, (collections.abc.Callable, type(None)))
        ]
        tensors.update(_get_tensors(members, seen))

    return tensors


def get_tensors(object_):
    """ Get all tensors associated with ``object_``

    Args:
        object_ (any): Any object to look for tensors.

    Returns:
        (list of torch.tensor): List of tensors that are associated with ``object_``.
    """
    return _get_tensors(object_)


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


def flatten_parameters(model):
    """ ``flatten_parameters`` of a RNN model loaded from disk. """
    model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)


def split_list(list_, splits):
    """ Split ``list_`` using the ``splits`` ratio.

    Args:
        list_ (list): List to split.
        splits (tuple): Tuple of decimals determining list splits summing up to 1.0.

    Returns:
        (list): Splits of the list.

    Example:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> split_list(dataset, splits=(.6, .2, .2))
        [[1, 2, 3], [4], [5]]
    """
    assert sum(splits) == 1, 'Splits must sum to 1.0'
    splits = [round(s * len(list_)) for s in splits]
    lists = []
    for split in splits[:-1]:
        lists.append(list_[:split])
        list_ = list_[split:]
    lists.append(list_)
    return lists


def get_total_parameters(model):
    """ Return the total number of trainable parameters in ``model``.

    Args:
        model (torch.nn.Module)

    Returns:
        (int): The total number of trainable parameters in ``model``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])
        >>> lengths_to_mask([1, 2, 2], [1, 2, 2])
        tensor([[[ True, False],
                 [False, False]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]]])

    Args:
        *lengths (list of int or torch.Tensor)
        **kwargs: Keyword arguments passed to ``torch.zeros`` upon initially creating the returned
          tensor.

    Returns:
        torch.BoolTensor
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
    return mask.bool()


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


def identity(x):
    return x
