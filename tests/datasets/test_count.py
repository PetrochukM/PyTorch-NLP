import pytest

from torchnlp.datasets import count_dataset


def test_count_dataset():
    test_data = count_dataset(test=True)
    train_data, dev_data, test_data = count_dataset(test=True, train=True, dev=True)
    # Test if data generated is consistent
    assert list(test_data) == list(test_data)
    assert len(train_data) > 0
    assert len(dev_data) > 0
    assert len(test_data) > 0


def test_count_dataset_rows():
    test_data = count_dataset(test=True, test_rows=100)
    assert len(test_data) == 100


def test_count_dataset_column():
    test_data = count_dataset(test=True, test_rows=100)
    assert len(test_data['numbers']) == 100
    assert len(test_data['count']) == 100

    # Column does not exist
    with pytest.raises(AttributeError):
        test_data['text']
