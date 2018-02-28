import pytest

from torchnlp.datasets import count_dataset
from tests.datasets.utils import try_dataset


def test_count_dataset():
    # NOTE: Just make sure it works...
    try_dataset(count_dataset)


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
