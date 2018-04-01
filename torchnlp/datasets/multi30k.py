import os

from torchnlp.utils import download_urls
from torchnlp.datasets.dataset import Dataset


def multi30k_dataset(directory='data/multi30k/',
                     train=False,
                     dev=False,
                     test=False,
                     language_extensions=['en', 'de'],
                     train_filename='train',
                     dev_filename='val',
                     test_filename='test',
                     check_file='train.de',
                     urls=[
                         'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
                         'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
                         'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz'
                     ]):
    """
    Load the WMT 2016 machine translation dataset.

    As a translation task, this task consists in translating English sentences that describe an
    image into German, given the English sentence itself. As training and development data, we
    provide 29,000 and 1,014 triples respectively, each containing an English source sentence, its
    German human translation. As test data, we provide a new set of 1,000 tuples containing an
    English description.

    More details:
    http://www.statmt.org/wmt16/multimodal-task.html
    http://shannon.cs.illinois.edu/DenotationGraph/

    Citation:
    ```
        @article{elliott-EtAl:2016:VL16,
            author    = {{Elliott}, D. and {Frank}, S. and {Sima'an}, K. and {Specia}, L.},
            title     = {Multi30K: Multilingual English-German Image Descriptions},
            booktitle = {Proceedings of the 5th Workshop on Vision and Language},
            year      = {2016},
            pages     = {70--74},
            year      = 2016
        }
    ```

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        language_extensions (:class:`list` of :class:`str`): List of language extensions ['en'|'de']
            to load.
        train_directory (str, optional): The directory of the training split.
        dev_directory (str, optional): The directory of the dev split.
        test_directory (str, optional): The directory of the test split.
        check_file (str, optional): Check this file exists if download was successful.
        urls (str, optional): URLs to download.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training tokens, dev
        tokens and test tokens in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import multi30k_dataset
        >>> train = multi30k_dataset(train=True)
        >>> train[:2]
        [{
          'en': 'Two young, White males are outside near many bushes.',
          'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'
        }, {
          'en': 'Several men in hard hatsare operating a giant pulley system.',
          'de': 'Mehrere Männer mit Schutzhelmen bedienen ein Antriebsradsystem.'
        }]
    """
    download_urls(directory=directory, file_urls=urls, check_file=check_file)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        examples = []
        for extension in language_extensions:
            path = os.path.join(directory, filename + '.' + extension)
            with open(path, 'r', encoding='utf-8') as f:
                language_specific_examples = [l.strip() for l in f]

            if len(examples) == 0:
                examples = [{} for _ in range(len(language_specific_examples))]
            for i, example in enumerate(language_specific_examples):
                examples[i][extension] = example

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
