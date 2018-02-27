import pytest
import numpy as np
import os
import shutil

from lib.pretrained_embeddings import FastText
from lib.pretrained_embeddings import CharNGram
from lib.pretrained_embeddings import GloVe

# TODO: Consider making the tests faster by mocking the download

directory = os.path.dirname(os.path.realpath(__file__))
cache = os.path.join(directory, '.cache')


@pytest.mark.slow
@pytest.mark.capture_disabled
def test_charngram_100d():
    vectors = CharNGram(cache=cache)
    # Test vector dim
    assert len(vectors['hi']) == 100
    shutil.rmtree(cache)


@pytest.mark.slow
@pytest.mark.capture_disabled
def test_fasttext_simple():
    vectors = FastText(language="simple", cache=cache)
    assert len(vectors['hi']) == 300

    # Test cache and is_include
    vectors = FastText(language="simple", is_include=lambda w: w == 'hi')
    assert len(vectors) == 1
    shutil.rmtree(cache)


@pytest.mark.slow
@pytest.mark.capture_disabled
def test_glove_6b_50():
    vectors = GloVe(name="6B", dim="50", cache=cache)
    # Test vector dim
    assert len(vectors['hi']) == 50
    shutil.rmtree(cache)


@pytest.mark.slow
@pytest.mark.capture_disabled
def test_glove_twitter_6b_25():
    vectors = GloVe(name="twitter.27B", dim="25", cache=cache)
    np.testing.assert_array_almost_equal(
        vectors['hi'].tolist(), [
            -0.1792, 0.1028, -0.1085, -0.0724, -1.2424, -1.0749, 0.5523, 1.1216, -0.5209, 0.4347,
            -0.7169, -0.3777, -3.2797, -0.0685, -0.7120, -0.2615, 0.6470, -0.3853, -1.2116, 0.3721,
            0.1887, 0.3030, -0.2902, 0.8507, -1.3869
        ],
        decimal=3)
    shutil.rmtree(cache)
