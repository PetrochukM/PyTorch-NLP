import pytest

from lib.datasets import simple_qa_dataset
from tests.datasets.utils import try_dataset


@pytest.mark.slow
def test_simple_qa_dataset_dataset():
    # NOTE: Just make sure it works...
    try_dataset(simple_qa_dataset)


@pytest.mark.slow
def test_simple_qa_dataset_row():
    # Load the smallest split ~ 10,000 rows
    dev = simple_qa_dataset(dev=True)
    assert dev[0] == {
        'question': 'Who was the trump ocean club international hotel and tower named after',
        'relation': 'www.freebase.com/symbols/namesake/named_after',
        'object': 'www.freebase.com/m/0cqt90',
        'subject': 'www.freebase.com/m/0f3xg_'
    }
