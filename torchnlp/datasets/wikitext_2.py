import os
import io

from torchnlp.download import download_file_maybe_extract
from torchnlp.text_encoders import UNKNOWN_TOKEN
from torchnlp.text_encoders import EOS_TOKEN


def wikitext_2_dataset(
        directory='data/',
        train=False,
        dev=False,
        test=False,
        train_filename='wiki.train.tokens',
        dev_filename='wiki.valid.tokens',
        test_filename='wiki.test.tokens',
        extracted_name='wikitext-2',
        check_files=['wikitext-2/wiki.train.tokens'],
        url='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'):
    """
    Load the WikiText-2 dataset.

    The WikiText language modeling dataset is a collection of over 100 million tokens extracted
    from the set of verified Good and Featured articles on Wikipedia. The dataset is available
    under the Creative Commons Attribution-ShareAlike License.

    **Reference:**
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

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
        >>> from torchnlp.datasets import wikitext_2_dataset
        >>> train = wikitext_2_dataset(train=True)
        >>> train[:10]
        ['</s>', '=', 'Valkyria', 'Chronicles', 'III', '=', '</s>', '</s>', 'Senjō', 'no']
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, extracted_name, filename)
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
