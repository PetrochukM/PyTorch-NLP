import random

from torchnlp.datasets.dataset import Dataset


def count_dataset(train=False,
                  dev=False,
                  test=False,
                  train_rows=10000,
                  dev_rows=1000,
                  test_rows=1000,
                  seq_max_length=10):
    """
    Load the Count dataset.

    The Count dataset is a simple task of counting the number of integers in a sequence. This
    dataset is useful for testing implementations of sequence to label models.

    Args:
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_rows (int, optional): Number of training rows to generate.
        dev_rows (int, optional): Number of development rows to generate.
        test_rows (int, optional): Number of test rows to generate.
        seq_max_length (int, optional): Maximum sequence length.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> import random
        >>> random.seed(321)
        >>>
        >>> from torchnlp.datasets import count_dataset
        >>> train = count_dataset(train=True)
        >>> train[0:2]
        [{'numbers': '6 2 5 8 7', 'count': '5'}, {'numbers': '3 9 7 6 6 7', 'count': '6'}]
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
            rows.append({'numbers': input_, 'count': str(length)})

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
