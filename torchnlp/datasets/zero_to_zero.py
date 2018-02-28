from torchnlp.datasets.dataset import Dataset


def zero_to_zero_dataset(train=False,
                         dev=False,
                         test=False,
                         train_rows=256,
                         dev_rows=64,
                         test_rows=64):
    """
    Used for quick integration testing.

    Sample Data:
        Input: 0
        Output: 0
    """
    ret = []
    for is_requested, n_rows in [(train, train_rows), (dev, dev_rows), (test, test_rows)]:
        if not is_requested:
            continue
        rows = [{'source': str(0), 'target': str(0)} for i in range(n_rows)]
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
