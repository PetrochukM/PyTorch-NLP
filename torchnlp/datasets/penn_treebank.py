import os
import io
import urllib.request

from tqdm import tqdm

from torchnlp.utils import reporthook
from torchnlp.text_encoders import UNKNOWN_TOKEN
from torchnlp.text_encoders import EOS_TOKEN


def _download_penn_treebank_dataset(directory, name, urls, check_file):
    """ Download the penn treebank dataset """
    # Already downloaded
    if check_file is not None and os.path.isfile(os.path.join(directory, check_file)):
        return

    dataset_directory = os.path.join(directory, name)
    if not os.path.isdir(dataset_directory):
        os.makedirs(dataset_directory)

    for url in urls:
        basename = os.path.basename(url)
        filename = os.path.join(dataset_directory, basename)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=basename) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=reporthook(t))

    if check_file is not None and not os.path.isfile(os.path.join(directory, check_file)):
        raise ValueError('[DOWNLOAD FAILED] `check_file` not found')


def penn_treebank_dataset(
        directory='data/',
        train=False,
        dev=False,
        test=False,
        train_filename='ptb.train.txt',
        dev_filename='ptb.valid.txt',
        test_filename='ptb.test.txt',
        name='penn-treebank',
        check_file='penn-treebank/ptb.train.txt',
        urls=[
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
        ]):
    """
    Load the Penn Treebank dataset.

    This is the Penn Treebank Project: Release 2 CDROM, featuring a million words of 1989 Wall
    Street Journal material.

    More details:
    https://catalog.ldc.upenn.edu/ldc99t42

    Citation:
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the development split.
        test_filename (str, optional): The filename of the test split.
        name (str, optional): Name of the dataset directory.
        check_file (str, optional): Check this file exists if download was successful.
        urls (str, optional): URLs of the dataset `tar.gz` file.

    Returns:
        :class:`tuple` of :class:`list` of :class:`str`: Tuple with the training tokens, dev tokens
        and test tokens in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import penn_treebank_dataset
        >>> train = penn_treebank_dataset(train=True)
        >>> train[:10]
        ['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano',
        'guterman', 'hydro-quebec']
    """
    _download_penn_treebank_dataset(
        directory=directory, name=name, urls=urls, check_file=check_file)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    split_filenames = [dir_ for (requested, dir_) in splits if requested]
    for filename in split_filenames:
        full_path = os.path.join(directory, name, filename)
        text = []
        with io.open(full_path, encoding='utf-8') as f:
            for line in f:
                text.extend(line.replace('<unk>', UNKNOWN_TOKEN).split())
                text.append(EOS_TOKEN)
        ret.append(text)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
