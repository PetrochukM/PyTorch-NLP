import os

import mock

from torchnlp.datasets import multi30k_dataset
from tests.datasets.utils import urlretrieve_side_effect

multi30k_directory = 'tests/_test_data/multi30k'


@mock.patch("urllib.request.urlretrieve")
def test_multi30k_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = multi30k_dataset(
        directory=multi30k_directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
        'en': 'Two young, White males are outside near many bushes.'
    }

    # Clean up
    os.remove(os.path.join(multi30k_directory, 'train.en'))
    os.remove(os.path.join(multi30k_directory, 'train.de'))
    os.remove(os.path.join(multi30k_directory, 'test.en'))
    os.remove(os.path.join(multi30k_directory, 'test.de'))
    os.remove(os.path.join(multi30k_directory, 'val.en'))
    os.remove(os.path.join(multi30k_directory, 'val.de'))
