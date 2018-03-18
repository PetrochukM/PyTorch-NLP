import urllib.request
import os
import shutil

import mock

from torchnlp.datasets import simple_qa_dataset

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_simple_qa_dataset_row(mock_urlretrieve):
    # Check the URL requested is valid
    def side_effect(url, **kwargs):
        # TODO: Fix failure case if internet does not work
        assert urllib.request.urlopen(url).getcode() == 200

    mock_urlretrieve.side_effect = side_effect

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
