import os
import shutil

import mock

from torchnlp.datasets import snli_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_snli_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = snli_dataset(directory=directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'premise':
            'A person on a horse jumps over a broken down airplane.',
        'hypothesis':
            'A person is training his horse for a competition.',
        'label':
            'neutral',
        'premise_transitions': [
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'reduce', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'reduce', 'shift', 'reduce', 'shift', 'reduce', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'reduce', 'shift', 'reduce', 'shift', 'reduce', 'shift', 'reduce', 'shift',
            'reduce', 'shift', 'shift', 'shift', 'reduce', 'shift', 'reduce'
        ],
        'hypothesis_transitions': [
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'reduce', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'reduce', 'shift', 'reduce', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift', 'shift',
            'shift', 'reduce', 'shift', 'reduce', 'shift', 'reduce', 'shift', 'reduce', 'shift',
            'shift', 'shift', 'reduce', 'shift', 'reduce'
        ]
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'snli_1.0'))
