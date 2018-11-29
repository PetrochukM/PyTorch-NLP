import os
import io

from torchnlp.download import download_file_maybe_extract
from torchnlp.datasets.dataset import Dataset


def get_label_str(label, fine_grained=False):
    pre = 'very ' if fine_grained else ''
    return {
        '0': pre + 'negative',
        '1': 'negative',
        '2': 'neutral',
        '3': 'positive',
        '4': pre + 'positive',
        None: None
    }[label]


def parse_tree(data, subtrees=False, fine_grained=False):
    # https://github.com/pytorch/text/blob/6476392a801f51794c90378dd23489578896c6f2/torchtext/data/example.py#L56
    try:
        from nltk.tree import Tree
    except ImportError:
        print("Please install NLTK. " "See the docs at http://nltk.org for more information.")
        raise
    tree = Tree.fromstring(data)

    if subtrees:
        return [{
            'text': ' '.join(t.leaves()),
            'label': get_label_str(t.label(), fine_grained=fine_grained)
        } for t in tree.subtrees()]

    return {
        'text': ' '.join(tree.leaves()),
        'label': get_label_str(tree.label(), fine_grained=fine_grained)
    }


def smt_dataset(directory='data/',
                train=False,
                dev=False,
                test=False,
                train_filename='train.txt',
                dev_filename='dev.txt',
                test_filename='test.txt',
                extracted_name='trees',
                check_files=['trees/train.txt'],
                url='http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip',
                fine_grained=False,
                subtrees=False):
    """
    Load the Stanford Sentiment Treebank dataset.

    Semantic word spaces have been very useful but cannot express the meaning of longer phrases in
    a principled way. Further progress towards understanding compositionality in tasks such as
    sentiment detection requires richer supervised training and evaluation resources and more
    powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes
    fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and
    presents new challenges for sentiment compositionality.

    **Reference**:
    https://nlp.stanford.edu/sentiment/index.html

    **Citation:**
    Richard Socher, Alex Perelygin, Jean Y. Wu, Jason Chuang, Christopher D. Manning,
    Andrew Y. Ng and Christopher Potts. Recursive Deep Models for Semantic Compositionality Over a
    Sentiment Treebank

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
        subtrees (bool, optional): Whether to include sentiment-tagged subphrases in addition to
            complete examples.
        fine_grained (bool, optional): Whether to use 5-class instead of 3-class labeling.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training tokens, dev
        tokens and test tokens in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import smt_dataset
        >>> train = smt_dataset(train=True)
        >>> train[5]
        {
          'text': "Whether or not you 're enlightened by any of Derrida 's lectures on ...",
          'label': 'positive'
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
            for line in f:
                line = line.strip()
                if subtrees:
                    examples.extend(parse_tree(line, subtrees=subtrees, fine_grained=fine_grained))
                else:
                    examples.append(parse_tree(line, subtrees=subtrees, fine_grained=fine_grained))
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
