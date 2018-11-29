import os
import io

from torchnlp.text_encoders import UNKNOWN_TOKEN
from torchnlp.text_encoders import EOS_TOKEN
from torchnlp.download import download_files_maybe_extract


def penn_treebank_dataset(
        directory='data/penn-treebank',
        train=False,
        dev=False,
        test=False,
        train_filename='ptb.train.txt',
        dev_filename='ptb.valid.txt',
        test_filename='ptb.test.txt',
        check_files=['ptb.train.txt'],
        urls=[
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
        ]):
    """
    Load the Penn Treebank dataset.

    This is the Penn Treebank Project: Release 2 CDROM, featuring a million words of 1989 Wall
    Street Journal material.

    **Reference:** https://catalog.ldc.upenn.edu/ldc99t42

    **Citation:**
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
        check_files (str, optional): Check if these files exist, then this download was successful.
        urls (str, optional): URLs to download.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> from torchnlp.datasets import penn_treebank_dataset
        >>> train = penn_treebank_dataset(train=True)
        >>> train[:10]
        ['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano',
        'guterman', 'hydro-quebec']
    """
    download_files_maybe_extract(urls=urls, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, filename)
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
