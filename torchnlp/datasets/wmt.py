import os

from torchnlp.download import download_file_maybe_extract
from torchnlp.datasets.dataset import Dataset


def wmt_dataset(directory='data/wmt16_en_de',
                train=False,
                dev=False,
                test=False,
                train_filename='train.tok.clean.bpe.32000',
                dev_filename='newstest2013.tok.bpe.32000',
                test_filename='newstest2014.tok.bpe.32000',
                check_files=['train.tok.clean.bpe.32000.en'],
                url='https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8'):
    """
    The Workshop on Machine Translation (WMT) 2014 English-German dataset.

    Initially this dataset was preprocessed by Google Brain. Though this download contains test sets
    from 2015 and 2016, the train set differs slightly from WMT 2015 and 2016 and significantly from
    WMT 2017.

    The provided data is mainly taken from version 7 of the Europarl corpus, which is freely
    available. Note that this the same data as last year, since Europarl is not anymore translted
    across all 23 official European languages. Additional training data is taken from the new News
    Commentary corpus. There are about 50 million words of training data per language from the
    Europarl corpus and 3 million words from the News Commentary corpus.

    A new data resource from 2013 is the Common Crawl corpus which was collected from web sources.
    Each parallel corpus comes with a annotation file that gives the source of each sentence pair.

    References:
        * https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/translate_ende.py # noqa: E501
        * http://www.statmt.org/wmt14/translation-task.html

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the dev split.
        test_filename (str, optional): The filename of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `tar.gz` file.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> from torchnlp.datasets import wmt_dataset
        >>> train = wmt_dataset(train=True)
        >>> train[:2]
        [{
          'en': 'Res@@ um@@ ption of the session',
          'de': 'Wiederaufnahme der Sitzungsperiode'
        }, {
          'en': 'I declare resumed the session of the European Parliament ad@@ jour@@ ned on...'
          'de': 'Ich erklär@@ e die am Freitag , dem 17. Dezember unterbro@@ ch@@ ene...'
        }]
    """
    download_file_maybe_extract(
        url=url, directory=directory, check_files=check_files, filename='wmt16_en_de.tar.gz')

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]

    for filename in splits:
        examples = []

        en_path = os.path.join(directory, filename + '.en')
        de_path = os.path.join(directory, filename + '.de')
        en_file = [l.strip() for l in open(en_path, 'r', encoding='utf-8')]
        de_file = [l.strip() for l in open(de_path, 'r', encoding='utf-8')]
        assert len(en_file) == len(de_file)
        for i in range(len(en_file)):
            if en_file[i] != '' and de_file[i] != '':
                examples.append({'en': en_file[i], 'de': de_file[i]})

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
