import os

import pandas as pd

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract


def simple_qa_dataset(directory='data/',
                      train=False,
                      dev=False,
                      test=False,
                      extracted_name='SimpleQuestions_v2',
                      train_filename='annotated_fb_data_train.txt',
                      dev_filename='annotated_fb_data_valid.txt',
                      test_filename='annotated_fb_data_test.txt',
                      check_files=['SimpleQuestions_v2/annotated_fb_data_train.txt'],
                      url='https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz?raw=1'):
    """
    Load the SimpleQuestions dataset.

    Single-relation factoid questions (simple questions) are common in many settings
    (e.g. Microsoft’s search query logs and WikiAnswers questions). The SimpleQuestions dataset is
    one of the most commonly used benchmarks for studying single-relation factoid questions.

    **Reference:**
    https://research.fb.com/publications/large-scale-simple-question-answering-with-memory-networks/

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the development split.
        test_filename (str, optional): The filename of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `tar.gz` file.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset
        , dev dataset and test dataset in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import simple_qa_dataset
        >>> train = simple_qa_dataset(train=True)
        SimpleQuestions_v2.tgz:  15%|▏| 62.3M/423M [00:09<00:41, 8.76MB/s]
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
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, extracted_name, filename)
        data = pd.read_table(
            full_path, header=None, names=['subject', 'relation', 'object', 'question'])
        ret.append(
            Dataset([{
                'question': row['question'],
                'relation': row['relation'],
                'object': row['object'],
                'subject': row['subject'],
            } for _, row in data.iterrows()]))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
