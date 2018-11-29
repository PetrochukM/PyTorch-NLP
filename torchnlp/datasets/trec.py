import os

from torchnlp.download import download_files_maybe_extract
from torchnlp.datasets.dataset import Dataset


def trec_dataset(directory='data/trec/',
                 train=False,
                 test=False,
                 train_filename='train_5500.label',
                 test_filename='TREC_10.label',
                 check_files=['train_5500.label'],
                 urls=[
                     'http://cogcomp.org/Data/QA/QC/train_5500.label',
                     'http://cogcomp.org/Data/QA/QC/TREC_10.label'
                 ],
                 fine_grained=False):
    """
    Load the Text REtrieval Conference (TREC) Question Classification dataset.

    TREC dataset contains 5500 labeled questions in training set and another 500 for test set. The
    dataset has 6 labels, 50 level-2 labels. Average length of each sentence is 10, vocabulary size
    of 8700.

    References:
        * https://nlp.stanford.edu/courses/cs224n/2004/may-steinberg-project.pdf
        * http://cogcomp.org/Data/QA/QC/
        * http://www.aclweb.org/anthology/C02-1150

    **Citation:**
    Xin Li, Dan Roth, Learning Question Classifiers. COLING'02, Aug., 2002.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        test_filename (str, optional): The filename of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        urls (str, optional): URLs to download.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> from torchnlp.datasets import trec_dataset
        >>> train = trec_dataset(train=True)
        >>> train[:2]
        [{
          'label': 'DESC',
          'text': 'How did serfdom develop in and then leave Russia ?'
        }, {
          'label': 'ENTY',
          'text': 'What films featured the character Popeye Doyle ?'
        }]
    """
    download_files_maybe_extract(urls=urls, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, filename)
        examples = []
        for line in open(full_path, 'rb'):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b'\xf0', b' ').strip().decode().partition(' ')
            label, _, label_fine = label.partition(':')
            if fine_grained:
                examples.append({'label': label_fine, 'text': text})
            else:
                examples.append({'label': label, 'text': text})
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
