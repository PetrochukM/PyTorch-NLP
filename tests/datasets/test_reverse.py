from lib.datasets import reverse_dataset
from tests.datasets.utils import try_dataset


def test_reverse_dataset():
    # NOTE: Just make sure it works...
    try_dataset(reverse_dataset)


def test_reverse_dataset_rows():
    test_data = reverse_dataset(test=True, test_rows=100)
    assert len(test_data) == 100
