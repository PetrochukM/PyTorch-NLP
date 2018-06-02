import mock

from torchnlp.datasets import trec_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/trec'


@mock.patch("urllib.request.urlretrieve")
def test_penn_treebank_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, test = trec_dataset(directory=directory, test=True, train=True, check_files=[])
    assert len(train) > 0
    assert len(test) > 0
    assert train[:2] == [{
        'label': 'DESC',
        'text': 'How did serfdom develop in and then leave Russia ?'
    }, {
        'label': 'ENTY',
        'text': 'What films featured the character Popeye Doyle ?'
    }]

    train = trec_dataset(directory=directory, train=True, check_files=[], fine_grained=True)
    assert train[:2] == [{
        'label': 'manner',
        'text': 'How did serfdom develop in and then leave Russia ?'
    }, {
        'label': 'cremat',
        'text': 'What films featured the character Popeye Doyle ?'
    }]
