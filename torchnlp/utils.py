import logging
import logging.config

import random
import torch

logger = logging.getLogger(__name__)

# TODO: Add back padding


def shuffle(dataset, random_seed=123):
    """ Shuffle dataset deterministically based on random_seed. """
    random.Random(random_seed).shuffle(dataset.rows)


def resplit_datasets(dataset, other_dataset, random_seed=None, split=None):
    """Deterministic shuffle and split algorithm.

    Given the same two datasets and the same `random_seed`, the split happens the same exact way
    every call.

    Args:
        dataset (lib.datasets.Dataset): First dataset.
        other_dataset (lib.datasets.Dataset): Another dataset.
        random_seed (int, optional): Seed to control the shuffle of both datasets.
        split (float, optional): If defined it is the percentage of rows that first dataset gets
            after split otherwise the original proportions are kept.

    Returns:
        dataset (lib.datasets.Dataset)
        other_dataset (lib.datasets.Dataset)
    """
    # Prevent circular dependency
    from torchnlp.datasets import Dataset

    concat = dataset.rows + other_dataset.rows
    # Reference:
    # https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
    # NOTE: Shuffle the same way every call of `shuffle_datasets` where the `random_seed` is given
    random.Random(random_seed).shuffle(concat)
    if split is None:
        return Dataset(concat[:len(dataset)]), Dataset(concat[len(dataset):])
    else:
        split = max(min(round(len(concat) * split), len(concat)), 0)
        return Dataset(concat[:split]), Dataset(concat[split:])


def torch_equals_ignore_index(tensor, tensor_other, ignore_index=None):
    """
    Compute torch.equals with the optional mask parameter.

    Args:
        ignore_index (int, optional): specifies a tensor1 index that is ignored
    Returns:
        (bool) iff target and prediction are equal
    """
    if ignore_index is not None:
        assert tensor.size() == tensor_other.size()
        mask_arr = tensor.ne(ignore_index)
        tensor = tensor.masked_select(mask_arr)
        tensor_other = tensor_other.masked_select(mask_arr)

    return torch.equal(tensor, tensor_other)


def reporthook(t):
    """ Shuffle dataset deterministically based on random_seed. """
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner
