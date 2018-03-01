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
    Dataset introduced by this paper:
    https://research.fb.com/publications/large-scale-simple-question-answering-with-memory-networks/

    Example:
        First row from the development `annotated_fb_data_valid` dataset::

            subject: Who was the trump ocean club international hotel and tower named after
            relation: www.freebase.com/symbols/namesake/named_after
            object: www.freebase.com/m/0cqt90
            question: www.freebase.com/m/0f3xg_
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
