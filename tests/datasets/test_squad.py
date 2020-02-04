import os
import shutil

import mock

from torchnlp.datasets import squad_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'

import urllib.request


@mock.patch("urllib.request.urlretrieve")
def test_squad_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev = squad_dataset(directory=directory, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert train[0:2] == [{
        'question': 'In what country is Normandy located?', 
        'answer': ['France', 'France', 'France', 'France']
        }, {
        'question': 'When were the Normans in Normandy?', 
        'answer': ['10th and 11th centuries', 'in the 10th and 11th centuries', 
        '10th and 11th centuries', '10th and 11th centuries']
        }]

    # Clean up
    shutil.rmtree(os.path.join(directory, 'squad'))