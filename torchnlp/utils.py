from urllib.parse import urlparse

import logging
import os
import tarfile
import urllib.request
import zipfile

import random
import torch

from tqdm import tqdm

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


def flatten_parameters(model):
    """ ``flatten_parameters`` of a RNN model loaded from disk. """
    model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)


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


def get_filename_from_url(url):
    """ Return a filename from a URL """
    parse = urlparse(url)
    return os.path.basename(parse.path)


def download_urls(urls, directory, check_file=None):
    """ Download a set of ``urls`` into a ``directory``.
    
    Args:
        urls (:class:`list` of :class:`str`): Set of urls to download.
        directory (str): Directory in which to download urls.
        check_file (str, optional): Operation was successful if this file exists.
    Returns:
        None:
    """
    # Already downloaded
    if check_file is not None and os.path.isfile(os.path.join(directory, check_file)):
        return

    if not os.path.isdir(directory):
        os.makedirs(directory)

    for url in urls:
        filename = get_filename_from_url(url)
        full_path = os.path.join(directory, filename)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=full_path, reporthook=reporthook(t))

    if check_file is not None and not os.path.isfile(os.path.join(directory, check_file)):
        raise ValueError('[DOWNLOAD FAILED] `check_file` not found')


def download_compressed_directory(url, directory, check_file=None):
    """ Download a ``tar.gz`` from ``url`` and extract into ``directory``.

    Args:
        url (str): Url of a compressed directory
        directory (str): Directory to extract ``tar.gz`` to.
        check_file (str, optional): Operation was successful if this file exists.
    Returns:
        None:
    """
    if check_file is not None and os.path.isfile(os.path.join(directory, check_file)):
        # Already downloaded
        return

    if not os.path.isdir(directory):
        os.makedirs(directory)

    filename = get_filename_from_url(url)
    full_path = os.path.join(directory, filename)
    logger.info('Downloading {}'.format(filename))
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=full_path, reporthook=reporthook(t))
    logger.info('Extracting {}'.format(filename))
    extension = filename.split('.')[1:]
    if 'zip' in extension:
        with zipfile.ZipFile(full_path, "r") as zip:
            zip.extractall(directory)
    elif 'tar' in extension or 'tgz' in extension:
        with tarfile.open(full_path, mode='r') as tar:
            tar.extractall(path=directory)
    else:
        raise ValueError('Extension not supported: {}'.format(extension))

    logger.info('Done with Extracting {}'.format(filename))

    if check_file is not None and not os.path.isfile(os.path.join(directory, check_file)):
        raise ValueError('[DOWNLOAD FAILED] `check_file` not found')
