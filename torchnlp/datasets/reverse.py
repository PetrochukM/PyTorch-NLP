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
    Used for sequence to sequence tests.

    Sample Data:
        Input: 1 2 3
        Output: 3 2 1
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
