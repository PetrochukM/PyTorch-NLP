from torchnlp.datasets import zero_dataset


def test_zero_dataset():
    test_data = zero_dataset(test=True)
    train_data, dev_data, test_data = zero_dataset(test=True, train=True, dev=True)
    # Test if data generated is consistent
    assert list(test_data) == list(test_data)
    assert len(train_data) > 0
    assert len(dev_data) > 0
    assert len(test_data) > 0


def test_zero_dataset_rows():
    test_data = zero_dataset(test=True, test_rows=100)
    assert len(test_data) == 100
