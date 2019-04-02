import os
import shutil

import mock
import pytest

from torchnlp.datasets import simple_qa_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@pytest.mark.skip(reason="Simple Questions dataset url sometimes returns 404.")
@mock.patch("urllib.request.urlretrieve")
def test_simple_qa_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = simple_qa_dataset(directory=directory, test=True, train=True, dev=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert dev[0] == {
        'question': 'Who was the trump ocean club international hotel and tower named after',
        'relation': 'www.freebase.com/symbols/namesake/named_after',
        'object': 'www.freebase.com/m/0cqt90',
        'subject': 'www.freebase.com/m/0f3xg_',
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'SimpleQuestions_v2'))
