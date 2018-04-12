import os
import mock

from torchnlp.word_to_vector import FastText
from tests.embeddings.utils import urlretrieve_side_effect


@mock.patch('urllib.request.urlretrieve')
def test_fasttext_simple(mock_urlretrieve):
    directory = 'tests/_test_data/fast_text/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of FastText
    vectors = FastText(language='simple', cache=directory)
    assert len(vectors['the']) == 300
    assert len(vectors) > 1

    # Test cache and `is_include`
    vectors = FastText(language='simple', is_include=lambda w: w == 'the', cache=directory)
    assert 'the' in vectors.stoi
    assert len(vectors) == 1

    # Test with the unknown characters
    assert len(vectors['漢字']) == 300

    # Clean up
    os.remove(os.path.join(directory, 'wiki.simple.vec.pt'))
