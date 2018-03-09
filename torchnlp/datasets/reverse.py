import random

from torchnlp.datasets.dataset import Dataset


def reverse_dataset(train=False,
                    dev=False,
                    test=False,
                    train_rows=10000,
                    dev_rows=1000,
                    test_rows=1000,
                    seq_max_length=10):
    """
    Load the Reverse dataset.

    The Reverse dataset is a simple task of reversing a list of numbers. This dataset is useful
    for testing implementations of sequence to sequence models.

    Args:
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_rows (int, optional): Number of training rows to generate.
        dev_rows (int, optional): Number of development rows to generate.
        test_rows (int, optional): Number of test rows to generate.
        seq_max_length (int, optional): Maximum sequence length.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset
        , dev dataset and test dataset in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import reverse_dataset
        >>> train = reverse_dataset(train=True)
        >>> train[0:2]
        [{
          'source': '5 2 7 5 1',
          'target': '1 5 7 2 5'
        }, {
          'source': '5 0 3 8 4 1 9 0 0 2',
          'target': '2 0 0 9 1 4 8 3 0 5'
        }]
    """
    ret = []
    for is_requested, n_rows in [(train, train_rows), (dev, dev_rows), (test, test_rows)]:
        rows = []
        for i in range(n_rows):
            length = random.randint(1, seq_max_length)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            input_ = ' '.join(seq)
            output = ' '.join(reversed(seq))
            rows.append({'source': input_, 'target': output})

        # NOTE: Given that `random.randint` is deterministic with the same `random_seed` we need
        # to allow the random generator to create the train, dev and test dataset in order.
        # Otherwise, `reverse(train=True)` and `reverse(test=True)` would share the first 1000 rows.
        if not is_requested:
            continue

        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
