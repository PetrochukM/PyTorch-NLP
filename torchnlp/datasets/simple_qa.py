import logging
import os
import tarfile
import urllib.request

import pandas as pd
from tqdm import tqdm

from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import reporthook

logger = logging.getLogger(__name__)


def _download_simple_qa_dataset(directory, some_file='annotated_fb_data_train.txt'):
    """ Download the Simple Questions dataset into `directory`

    Args:
        directory (str)
        some_file (str): Used to make sure Simple Questions was downloaded and extracted.
     """
    if os.path.isdir(directory) and os.path.isfile(os.path.join(directory, some_file)):
        # Already downloaded
        return

    if not os.path.isdir(directory):
        os.makedirs(directory)

    url = 'https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz?raw=1'
    filename = os.path.join(directory, 'SimpleQuestions_v2.tgz')
    logger.info('Downloading Simple Questions...')
    with tqdm() as t:  # all optional kwargs
        urllib.request.urlretrieve(url, filename=filename, reporthook=reporthook(t))
    logger.info('Extracting Simple Questions...')
    tarfile_ = tarfile.open(filename, mode='r')
    for member in tarfile_.getmembers():
        if member.isreg() and '._' != os.path.split(
                member.name)[1][:2]:  # skip if the TarInfo is not files and not hidden file
            member.name = os.path.normpath(member.name)
            # remove the root folder SimpleQuestions_v2
            member.name = os.path.join(*member.name.split('/')[1:])
            tarfile_.extract(member=member, path=directory)
    tarfile_.close()


def simple_qa_dataset(directory='data/simple_qa',
                      train=False,
                      dev=False,
                      test=False,
                      train_filename='annotated_fb_data_train.txt',
                      dev_filename='annotated_fb_data_valid.txt',
                      test_filename='annotated_fb_data_test.txt'):
    """
    Load the SimpleQuestions dataset.

    Single-relation factoid questions (simple questions) are common in many settings
    (e.g. Microsoft’s search query logs and WikiAnswers questions). The SimpleQuestions dataset is
    one of the most commonly used benchmarks for studying single-relation factoid questions.

    Paper introducing the dataset:
    https://research.fb.com/publications/large-scale-simple-question-answering-with-memory-networks/

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the development split.
        test_filename (str, optional): The filename of the test split.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset
        , dev dataset and test dataset in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import simple_qa_dataset
        >>> train = simple_qa_dataset(train=True)
        >>> train[0:2]
        [{
          'question': 'what is the book e about',
          'relation': 'www.freebase.com/book/written_work/subjects',
          'object': 'www.freebase.com/m/01cj3p',
          'subject': 'www.freebase.com/m/04whkz5'
        }, {
          'question': 'to what release does the release track cardiac arrest come from',
          'relation': 'www.freebase.com/music/release_track/release',
          'object': 'www.freebase.com/m/0sjc7c1',
          'subject': 'www.freebase.com/m/0tp2p24'
        }]
    """
    _download_simple_qa_dataset(directory, train_filename)

    ret = []
    datasets = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    for is_requested, filename in datasets:
        if not is_requested:
            continue
        full_path = os.path.join(directory, filename)
        data = pd.read_table(
            full_path, header=None, names=['subject', 'relation', 'object', 'question'])
        rows = []
        for _, row in data.iterrows():
            rows.append({
                'question': row['question'],
                'relation': row['relation'],
                'object': row['object'],
                'subject': row['subject'],
            })
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
