import os
import mock

from torchnlp.word_to_vector import FastText
from tests.word_to_vector.utils import urlretrieve_side_effect


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

    # Test implementation of __contains()__
    assert 'the' in vectors

    # Test with the unknown characters
    assert len(vectors['漢字']) == 300

    # Clean up
    os.remove(os.path.join(directory, 'wiki.simple.vec.pt'))


@mock.patch('urllib.request.urlretrieve')
def test_fasttext_list_arguments(mock_urlretrieve):
    directory = 'tests/_test_data/fast_text/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Load subset of FastText
    vectors = FastText(language='simple', cache=directory)

    # Test implementation of __getitem()__ for token list and tuple
    list(vectors[['the', 'of']].shape) == [2, 300]
    list(vectors[('the', 'of')].shape) == [2, 300]

    # Clean up
    os.remove(os.path.join(directory, 'wiki.simple.vec.pt'))


@mock.patch('urllib.request.urlretrieve')
def test_fasttext_non_list_or_tuple_raises_type_error(mock_urlretrieve):
    directory = 'tests/_test_data/fast_text/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Load subset of FastText
    vectors = FastText(language='simple', cache=directory)

    # Test implementation of __getitem()__ for invalid type
    error_class = None

    try:
        vectors[None]
    except Exception as e:
        error_class = e.__class__

    assert error_class is TypeError

    # Clean up
    os.remove(os.path.join(directory, 'wiki.simple.vec.pt'))


@mock.patch('urllib.request.urlretrieve')
def test_aligned_fasttext(mock_urlretrieve):
    directory = 'tests/_test_data/fast_text/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Parse the aligned FastText embeddings
    vectors = FastText(aligned=True, cache=directory)

    # Assert the embeddings' dimensionality
    assert len(vectors['the']) == 300
    # Our test file contains only five words to keep the file size small
    assert len(vectors) == 5

    # Clean up
    os.remove(os.path.join(directory, 'wiki.multi.en.vec.pt'))
