import mock

from torchnlp.datasets import penn_treebank_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/penn-treebank'


@mock.patch("urllib.request.urlretrieve")
def test_penn_treebank_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev, test = penn_treebank_dataset(
        directory=directory, test=True, dev=True, train=True, check_files=[])
    assert len(train) > 0
    assert len(test) > 0
    assert len(dev) > 0
    assert train[0:10] == [
        'aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano',
        'guterman', 'hydro-quebec'
    ]
