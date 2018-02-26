from lib.datasets import count_dataset
from tests.datasets.utils import try_dataset


def test_count_dataset():
    # NOTE: Just make sure it works...
    try_dataset(count_dataset)


def test_count_dataset_rows():
    test_data = count_dataset(test=True, test_rows=100)
    assert len(test_data) == 100
