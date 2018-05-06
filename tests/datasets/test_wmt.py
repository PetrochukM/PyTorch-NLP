import os

import mock

from torchnlp.datasets import wmt_dataset
from tests.datasets.utils import download_from_drive_side_effect

directory = 'tests/_test_data/wmt16_en_de'


@mock.patch('torchnlp.download._download_file_from_drive')
def test_wmt_dataset(mock_download_from_drive):
    mock_download_from_drive.side_effect = download_from_drive_side_effect

    # Check a row are parsed correctly
    train, dev, test = wmt_dataset(directory=directory, test=True, dev=True, train=True)
    assert len(train) > 0
    assert len(test) > 0
    assert len(dev) > 0
    assert train[0] == {
        'en': 'Res@@ um@@ ption of the session',
        'de': 'Wiederaufnahme der Sitzungsperiode'
    }

    # Clean up
    os.remove(os.path.join(directory, 'bpe.32000'))
    os.remove(os.path.join(directory, 'newstest2013.tok.bpe.32000.en'))
    os.remove(os.path.join(directory, 'newstest2013.tok.bpe.32000.de'))
    os.remove(os.path.join(directory, 'newstest2014.tok.bpe.32000.en'))
    os.remove(os.path.join(directory, 'newstest2014.tok.bpe.32000.de'))
    os.remove(os.path.join(directory, 'train.tok.clean.bpe.32000.de'))
    os.remove(os.path.join(directory, 'train.tok.clean.bpe.32000.en'))
    os.remove(os.path.join(directory, 'vocab.bpe.32000'))
