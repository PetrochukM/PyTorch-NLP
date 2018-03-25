import os
import shutil

import mock

from torchnlp.datasets import ud_pos_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_ud_pos_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = ud_pos_dataset(directory=directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'tokens': [
            'Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al',
            '-', 'Ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of',
            'Qaim', ',', 'near', 'the', 'Syrian', 'border', '.'
        ],
        'ud_tags': [
            'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'ADJ', 'NOUN', 'VERB', 'PROPN', 'PROPN', 'PROPN',
            'PUNCT', 'PROPN', 'PUNCT', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN',
            'ADP', 'PROPN', 'PUNCT', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT'
        ],
        'ptb_tags': [
            'NNP', 'HYPH', 'NNP', ':', 'JJ', 'NNS', 'VBD', 'NNP', 'NNP', 'NNP', 'HYPH', 'NNP', ',',
            'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'NNP', ',', 'IN', 'DT', 'JJ',
            'NN', '.'
        ]
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'en-ud-v2'))
