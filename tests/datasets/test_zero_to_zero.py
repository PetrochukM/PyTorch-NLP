from torchnlp.datasets import zero_to_zero_dataset
from tests.datasets.utils import try_dataset


def test_zero_to_zero_dataset():
    # NOTE: Just make sure it works...
    try_dataset(zero_to_zero_dataset)


def test_zero_to_zero_dataset_rows():
    test_data = zero_to_zero_dataset(test=True, test_rows=100)
    assert len(test_data) == 100
