import numpy as np

from torchnlp.metrics import get_rouge_n


def test_rouge_1_exclusive():
    hypotheses = ["the cat was found under the bed"]
    references = ["the cat was under the bed"]
    result = get_rouge_n(hypotheses, references, n=1, exclusive=True)
    precision = result['p']
    np.testing.assert_almost_equal(precision, 0.833, decimal=3)

def test_rouge_1_inclusive():
    hypotheses = ["the cat was found under the bed"]
    references = ["the cat was under the bed"]
    result = get_rouge_n(hypotheses, references, n=1, exclusive=False)
    precision = result['p']
    np.testing.assert_almost_equal(precision, 0.857, decimal=3)

def test_rouge_2_exclusive():
    hypotheses =["police killed the gunman"]
    references = ["police kill the gunman"]
    result = get_rouge_n(hypotheses, references, exclusive=True)
    recall = result['p']
    np.testing.assert_almost_equal(recall, 0.333, decimal=3)