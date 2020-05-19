import numpy as np

from torchnlp.metrics import get_rouge_l_summary_level, get_rouge_n, get_rouge_w


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
    hypotheses = ["police killed the gunman"]
    references = ["police kill the gunman"]
    result = get_rouge_n(hypotheses, references, exclusive=True)
    recall = result['p']
    np.testing.assert_almost_equal(recall, 0.333, decimal=3)


def test_rouge_l_summary_level():
    hypotheses1 = ["police killed the gunman who injured 3 on campus"]
    hypotheses2 = ["The police was killed and the gunman ran off"]
    references = ["police killed the gunman and sealed off the scene"]
    result1 = get_rouge_l_summary_level(hypotheses1, references)
    result2 = get_rouge_l_summary_level(hypotheses2, references)
    f_measure1 = result1['f']
    f_measure2 = result2['f']
    np.testing.assert_almost_equal(f_measure1, 0.467, decimal=3)
    np.testing.assert_almost_equal(f_measure2, 0.584, decimal=3)


def test_rouge_w():
    hypotheses1 = "police killed the gunman who injured 3 on campus"
    hypotheses2 = "The police was killed and the gunman ran off"
    references = "police killed the gunman and sealed off the scene"
    result1 = get_rouge_w(hypotheses1, references)
    result2 = get_rouge_w(hypotheses2, references)
    f_measure1 = result1['f']
    f_measure2 = result2['f']
    np.testing.assert_almost_equal(f_measure1, 0.444, decimal=3)
    np.testing.assert_almost_equal(f_measure2, 0.294, decimal=3)
