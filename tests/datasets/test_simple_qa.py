import urllib.request
import os

import mock

from torchnlp.datasets import simple_qa_dataset
from tests.datasets.utils import try_dataset

directory = 'tests/_test_data/simple_qa'


@mock.patch("urllib.request.urlretrieve")
def test_simple_qa_dataset_row(mock_urlretrieve):
    # Check the URL requested is valid
    def side_effect(url, **kwargs):
        assert urllib.request.urlopen(url).getcode() == 200

    mock_urlretrieve.side_effect = side_effect

    # Try some basic stuff
    try_dataset(simple_qa_dataset)

    # Check a row are parsed correctly
    dev = simple_qa_dataset(directory=directory, dev=True)
    assert dev[0] == {
        'question': 'Who was the trump ocean club international hotel and tower named after',
        'relation': 'www.freebase.com/symbols/namesake/named_after',
        'object': 'www.freebase.com/m/0cqt90',
        'subject': 'www.freebase.com/m/0f3xg_',
    }

    # Clean up
    os.remove(os.path.join(directory, 'annotated_fb_data_test.txt'))
    os.remove(os.path.join(directory, 'annotated_fb_data_train.txt'))
    os.remove(os.path.join(directory, 'annotated_fb_data_valid.txt'))
