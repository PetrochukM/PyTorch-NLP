import urllib.request

from tqdm import tqdm

from torchnlp.download import _get_filename_from_url
from torchnlp.download import _reporthook


def test_get_filename_from_url():
    assert 'aclImdb_v1.tar.gz' in _get_filename_from_url(
        'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    assert 'SimpleQuestions_v2.tgz' in _get_filename_from_url(
        'https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz?raw=1')


def test_reporthook():
    # Check that reporthook works with URLLIB
    with tqdm(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve('http://google.com', reporthook=_reporthook(t))
