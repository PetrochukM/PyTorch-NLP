import os
import mock
import urllib.request

from torchnlp.pretrained_embeddings import FastText
from torchnlp.pretrained_embeddings import CharNGram
from torchnlp.pretrained_embeddings import GloVe


# Check the URL requested is valid
def urlretrieve_side_effect(url, **kwargs):
    assert urllib.request.urlopen(url).getcode() == 200


@mock.patch("urllib.request.urlretrieve")
def test_charngram_100d(mock_urlretrieve):
    directory = 'tests/_test_data/char_n_gram/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of CharNGram
    vectors = CharNGram(cache=directory)
    assert len(vectors['e']) == 100

    # Clean up
    os.remove(directory + 'charNgram.txt.pt')
    os.remove(directory + 'charNgram.txt')


@mock.patch("urllib.request.urlretrieve")
def test_fasttext_simple(mock_urlretrieve):
    directory = 'tests/_test_data/fast_text/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of FastText
    vectors = FastText(language="simple", cache=directory)
    assert len(vectors['the']) == 300
    assert len(vectors) > 1

    # Test cache and `is_include`
    vectors = FastText(language="simple", is_include=lambda w: w == 'the', cache=directory)
    assert 'the' in vectors.stoi
    assert len(vectors) == 1

    # Clean up
    os.remove(directory + 'wiki.simple.vec.pt')


@mock.patch("urllib.request.urlretrieve")
def test_glove_6b_50(mock_urlretrieve):
    directory = 'tests/_test_data/glove/'

    # Make sure URL has a 200 status
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Attempt to parse a subset of GloVe
    vectors = GloVe(name="6B", dim="50", cache=directory)
    assert len(vectors['the']) == 50

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

    # Clean up
    os.remove(directory + 'glove.twitter.27B.25d.txt.pt')
