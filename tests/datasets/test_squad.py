import os
import shutil

import mock

from torchnlp.datasets import squad_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_squad_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev = squad_dataset(directory=directory, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0

    assert len(train) == 130319
    assert len(dev) == 11873

    assert train[5] == {
        'question': 'In what R&B group was she the lead singer?',
        'answer': ["Destiny's Child"]
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'trees'))
