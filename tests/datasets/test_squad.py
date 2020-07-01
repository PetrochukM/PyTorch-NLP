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

    assert len(train) == 2
    assert len(dev) == 2

    assert train[0]['paragraphs'][0]['qas'][0]['question'] == (
        'When did Beyonce start becoming popular?')
    assert train[0]['paragraphs'][0]['qas'][0]['answers'] == [{
        'text': 'in the late 1990s',
        'answer_start': 269
    }]
