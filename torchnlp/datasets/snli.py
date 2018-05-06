import os
import io

import ujson as json

from torchnlp.download import download_file_maybe_extract
from torchnlp.datasets.dataset import Dataset


def snli_dataset(directory='data/',
                 train=False,
                 dev=False,
                 test=False,
                 train_filename='snli_1.0_train.jsonl',
                 dev_filename='snli_1.0_dev.jsonl',
                 test_filename='snli_1.0_test.jsonl',
                 extracted_name='snli_1.0',
                 check_files=['snli_1.0/snli_1.0_train.jsonl'],
                 url='http://nlp.stanford.edu/projects/snli/snli_1.0.zip'):
    """
    Load the Stanford Natural Language Inference (SNLI) dataset.

    The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs
    manually labeled for balanced classification with the labels entailment, contradiction, and
    neutral, supporting the task of natural language inference (NLI), also known as recognizing
    textual entailment (RTE). We aim for it to serve both as a benchmark for evaluating
    representational systems for text, especially including those induced by representation
    learning methods, as well as a resource for developing NLP models of any kind.

    **Reference:** https://nlp.stanford.edu/projects/snli/

    **Citation:**
    Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large
    annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference
    on Empirical Methods in Natural Language Processing (EMNLP).

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
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training tokens, dev
        tokens and test tokens in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import snli_dataset
        >>> train = snli_dataset(train=True)
        >>> train[0]
        {
          'premise': 'Kids are on a amusement ride.',
          'hypothesis': 'A car is broke down on the side of the road.',
          'label': 'contradiction',
          'premise_transitions': ['shift', 'shift', 'shift', 'shift', 'shift', 'shift', ...],
          'hypothesis_transitions': ['shift', 'shift', 'shift', 'shift', 'shift', 'shift', ...],
        }
    """
    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    get_transitions = lambda parse: ['reduce' if t == ')' else 'shift' for t in parse if t != '(']
    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, extracted_name, filename)
        examples = []
        with io.open(full_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = json.loads(line)
                examples.append({
                    'premise': line['sentence1'],
                    'hypothesis': line['sentence2'],
                    'label': line['gold_label'],
                    'premise_transitions': get_transitions(line['sentence1_binary_parse']),
                    'hypothesis_transitions': get_transitions(line['sentence2_binary_parse'])
                })
        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
