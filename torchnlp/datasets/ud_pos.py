import os
import io

from torchnlp.download import download_file_maybe_extract
from torchnlp.datasets.dataset import Dataset


def ud_pos_dataset(directory='data/',
                   train=False,
                   dev=False,
                   test=False,
                   train_filename='en-ud-tag.v2.train.txt',
                   dev_filename='en-ud-tag.v2.dev.txt',
                   test_filename='en-ud-tag.v2.test.txt',
                   extracted_name='en-ud-v2',
                   check_files=['en-ud-v2/en-ud-tag.v2.train.txt'],
                   url='https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip'):
    """
    Load the Universal Dependencies - English Dependency Treebank dataset.

    Corpus of sentences annotated using Universal Dependencies annotation. The corpus comprises
    254,830 words and 16,622 sentences, taken from various web media including weblogs, newsgroups,
    emails, reviews, and Yahoo! answers.

    References:
        * http://universaldependencies.org/
        * https://github.com/UniversalDependencies/UD_English

    **Citation:**
    Natalia Silveira and Timothy Dozat and Marie-Catherine de Marneffe and Samuel Bowman and
    Miriam Connor and John Bauer and Christopher D. Manning (2014).
    A Gold Standard Dependency Corpus for {E}nglish

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the development split.
        test_filename (str, optional): The filename of the test split.
        extracted_name (str, optional): Name of the extracted dataset directory.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `tar.gz` file.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> from torchnlp.datasets import ud_pos_dataset
        >>> train = ud_pos_dataset(train=True)
        >>> train[17] # Sentence at index 17 is shortish
        {
          'tokens': ['Guerrillas', 'killed', 'an', 'engineer', ',', 'Asi', 'Ali', ',', 'from',
                     'Tikrit', '.'],
          'ud_tags': ['NOUN', 'VERB', 'DET', 'NOUN', 'PUNCT', 'PROPN', 'PROPN', 'PUNCT', 'ADP',
                      'PROPN', 'PUNCT'],
          'ptb_tags': ['NNS', 'VBD', 'DT', 'NN', ',', 'NNP', 'NNP', ',', 'IN', 'NNP', '.']
        }
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, extracted_name, filename)
        examples = []
        with io.open(full_path, encoding='utf-8') as f:
            sentence = {'tokens': [], 'ud_tags': [], 'ptb_tags': []}
            for line in f:
                line = line.strip()
                if line == '' and len(sentence['tokens']) > 0:
                    examples.append(sentence)
                    sentence = {'tokens': [], 'ud_tags': [], 'ptb_tags': []}
                elif line != '':
                    token, ud_tag, ptb_tag = tuple(line.split('\t'))
                    sentence['tokens'].append(token)
                    sentence['ud_tags'].append(ud_tag)
                    sentence['ptb_tags'].append(ptb_tag)
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
