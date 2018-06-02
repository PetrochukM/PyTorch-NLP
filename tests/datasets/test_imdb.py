import os
import shutil

import mock

from torchnlp.datasets import imdb_dataset
from tests.datasets.utils import urlretrieve_side_effect

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_imdb_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, test = imdb_dataset(directory=directory, test=True, train=True)
    assert len(train) > 0
    assert len(test) > 0
    test = sorted(test, key=lambda r: len(r['text']))
    assert test[0] == {
        'text':
            "This movie was sadly under-promoted but proved to be truly exceptional. Entering " +
            "the theatre I knew nothing about the film except that a friend wanted to see it." +
            "<br /><br />I was caught off guard with the high quality of the film. I couldn't " +
            "image Ashton Kutcher in a serious role, but his performance truly exemplified his " +
            "character. This movie is exceptional and deserves our monetary support, unlike so " +
            "many other movies. It does not come lightly for me to recommend any movie, but in " +
            "this case I highly recommend that everyone see it.<br /><br />This films is Truly " +
            "Exceptional!",
        'sentiment':
            'pos'
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'aclImdb'))
