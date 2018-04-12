import os
import mock

from torchnlp.word_to_vector import GloVe
from tests.word_to_vector.utils import urlretrieve_side_effect


@mock.patch("urllib.request.urlretrieve")
def test_glove_6b_50(mock_urlretrieve):
    directory = 'tests/_test_data/glove/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of GloVe
    vectors = GloVe(name="6B", dim="50", cache=directory)
    assert len(vectors['the']) == 50

    # Test with the unknown characters
    assert len(vectors['漢字']) == 50

    # Clean up
    os.remove(directory + 'glove.6B.50d.txt.pt')


@mock.patch("urllib.request.urlretrieve")
def test_glove_twitter_6b_25(mock_urlretrieve):
    directory = 'tests/_test_data/glove/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of GloVe
    vectors = GloVe(name="twitter.27B", dim="25", cache=directory)
    assert len(vectors['the']) == 25

    # Test with the unknown characters
    assert len(vectors['漢字']) == 25

    # Clean up
    os.remove(directory + 'glove.twitter.27B.25d.txt.pt')
