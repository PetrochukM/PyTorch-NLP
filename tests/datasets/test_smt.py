import os
import shutil

import mock

from torchnlp.datasets import smt_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_smt_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = smt_dataset(directory=directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[5] == {
        'text':
            "Whether or not you 're enlightened by any of Derrida 's lectures on `` the other '' " +
            "and `` the self , '' Derrida is an undeniably fascinating and playful fellow .",
        'label':
            'positive'
    }
    train = smt_dataset(directory=directory, train=True, subtrees=True)
    assert train[3] == {'text': 'Rock', 'label': 'neutral'}

    train = smt_dataset(directory=directory, train=True, subtrees=True, fine_grained=True)
    assert train[4] == {
        'text':
            "is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a" +
            " splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven" +
            " Segal .",
        'label':
            'very positive'
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'trees'))
