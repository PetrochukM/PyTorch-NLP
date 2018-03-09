from torchnlp.datasets.dataset import Dataset


def zero_dataset(train=False, dev=False, test=False, train_rows=256, dev_rows=64, test_rows=64):
    """
    Load the Zero dataset.

    The Zero dataset is a simple task of predicting zero from zero. This dataset is useful for
    integration testing. The extreme simplicity of the dataset allows for models to learn the task
    quickly allowing for quick end-to-end testing.

    Args:
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the development split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        train_rows (int, optional): Number of training rows to generate.
        dev_rows (int, optional): Number of development rows to generate.
        test_rows (int, optional): Number of test rows to generate.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training dataset
        , dev dataset and test dataset in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import zero_dataset
        >>> train = zero_dataset(train=True)
        >>> train[0:2]
        [{
          'source': '0',
          'target': '0'
        }, {
          'source': '0',
          'target': '0'
        }]
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
