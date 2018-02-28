import shutil

import pytest

from torchnlp.datasets import simple_qa_dataset
from tests.datasets.utils import try_dataset

directory = 'tests/data/simple_qa'

# TODO: Consider mocking the simple_qa dataset for faster tests


@pytest.mark.slow
def test_simple_qa_dataset_dataset(directory=directory):
    # NOTE: Just make sure it works...
    try_dataset(simple_qa_dataset)
    shutil.rmtree(directory)


@pytest.mark.slow
def test_simple_qa_dataset_row():
    # Load the smallest split ~ 10,000 rows
    dev = simple_qa_dataset(directory=directory, dev=True)
    assert dev[0] == {
        'question': 'Who was the trump ocean club international hotel and tower named after',
        'relation': 'www.freebase.com/symbols/namesake/named_after',
        'object': 'www.freebase.com/m/0cqt90',
        'subject': 'www.freebase.com/m/0f3xg_'
    }

    shutil.rmtree(directory)
