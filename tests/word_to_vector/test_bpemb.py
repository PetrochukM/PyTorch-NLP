import os
import mock

from tests.word_to_vector.utils import urlretrieve_side_effect
from torchnlp.word_to_vector import BPEmb


@mock.patch('urllib.request.urlretrieve')
def test_bpemb(mock_urlretrieve):
    directory = 'tests/_test_data/bpemb/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of BPEmb
    vectors = BPEmb(cache=directory)

    # Our test data only contains a subset of 5 tokens
    assert len(vectors) == 5

    # Embedding dimensionalty should be 300 by default
    assert len(vectors['▁the']) == 300

    # Test implementation of __contains()__
    assert '▁the' in vectors

    # Test with the unknown characters
    assert len(vectors['漢字']) == 300

    # Clean up
    os.remove(os.path.join(directory, 'en.wiki.bpe.op50000.d300.w2v.txt.pt'))


def test_unsupported_language():
    error_class = None
    error_message = ''

    try:
        BPEmb(language='python')
    except Exception as e:
        error_class = e.__class__
        error_message = str(e)

    assert error_class is ValueError
    assert error_message.startswith("Language 'python' not supported.")


def test_unsupported_dim():
    error_class = None
    error_message = ''

    try:
        BPEmb(dim=42)
    except Exception as e:
        error_class = e.__class__
        error_message = str(e)

    assert error_class is ValueError
    assert error_message.startswith("Embedding dimensionality of '42' not " "supported.")


def test_unsupported_merge_ops():
    error_class = None
    error_message = ''

    try:
        BPEmb(merge_ops=42)
    except Exception as e:
        error_class = e.__class__
        error_message = str(e)

    assert error_class is ValueError
    assert error_message.startswith("Number of '42' merge operations not " "supported.")
