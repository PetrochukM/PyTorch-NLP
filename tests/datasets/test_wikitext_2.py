import os
import shutil

import mock

from torchnlp.datasets import wikitext_2_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_wikitext_2_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = wikitext_2_dataset(directory=directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(test) > 0
    assert len(dev) > 0
    assert train[0:10] == [
        '</s>', '=', 'Valkyria', 'Chronicles', 'III', '=', '</s>', '</s>', 'Senjō', 'no'
    ]

    # Clean up
    shutil.rmtree(os.path.join(directory, 'wikitext-2'))
