import logging
import logging.config

import random
import torch

logger = logging.getLogger(__name__)


def pad_tensor(tensor, length, padding_index):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.

    Args:
        tensor (1D :class:`torch.LongTensor`): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int): Index to pad tensor with.
    Returns
        torch.LongTensor: Padded Tensor.
    """
    assert len(tensor.size()) == 1
    assert length >= len(tensor)
    n_padding = length - len(tensor)
    padding = torch.LongTensor(n_padding * [padding_index])
    return torch.cat((tensor, padding), 0)


def pad_batch(batch, padding_index):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of 1D :class:`torch.LongTensor`): Batch of tensors to pad.
        padding_index (int): Index to pad tensors with.
    Returns
        list of torch.LongTensor, list of int: Padded tensors and original lengths of tensors.
    """
    lengths = [len(row) for row in batch]
    max_len = max(lengths)
    padded = [pad_tensor(row, max_len, padding_index) for row in batch]
    return padded, lengths


def shuffle(list_, random_seed=123):
    """ Shuffle list deterministically based on ``random_seed``.

    Reference:
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


def reporthook(t):
    """ ``reporthook`` to use with ``urllib.request`` that prints the process of the download.

    Reference:
    https://github.com/tqdm/tqdm
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner
