import os
import shutil

import mock

from torchnlp.datasets import iwslt_dataset
from tests.datasets.utils import urlretrieve_side_effect

iwslt_directory = 'tests/_test_data/iwslt'


@mock.patch("urllib.request.urlretrieve")
def test_iwslt_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = iwslt_dataset(directory=iwslt_directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    train = sorted(train, key=lambda r: len(r['en']))
    assert train[0] == {'en': 'Thank you.', 'de': 'Danke.'}

    # Smoke test for iwslt_clean running twice
    train, dev, test = iwslt_dataset(directory=iwslt_directory, test=True, dev=True, train=True)
    train = sorted(train, key=lambda r: len(r['en']))
    assert train[0] == {'en': 'Thank you.', 'de': 'Danke.'}

    # Clean up
    shutil.rmtree(os.path.join(iwslt_directory, 'en-de'))
