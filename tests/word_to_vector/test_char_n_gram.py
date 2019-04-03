import os
import mock

from torchnlp.word_to_vector import CharNGram
from tests.word_to_vector.utils import urlretrieve_side_effect
from torchnlp.encoders.text import DEFAULT_UNKNOWN_TOKEN


@mock.patch("urllib.request.urlretrieve")
def test_charngram_100d(mock_urlretrieve):
    directory = 'tests/_test_data/char_n_gram/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of CharNGram
    vectors = CharNGram(cache=directory)
    assert len(vectors['e']) == 100

    # Test with the unknown characters
    assert len(vectors['漢字']) == 100
    assert len(vectors[DEFAULT_UNKNOWN_TOKEN]) == 100

    # Clean up
    os.remove(directory + 'charNgram.txt.pt')
    os.remove(directory + 'charNgram.txt')
