import numpy as np

from torchnlp.metrics import get_moses_multi_bleu


# TODO: Fix failure case if internet does not work
def test_get_moses_multi_bleu():
    hypotheses = ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"]
    references = [
        "The quick brown fox jumps over the lazy dog 笑",
        "The quick brown fox jumps over the lazy dog 笑"
    ]
    result = get_moses_multi_bleu(hypotheses, references, lowercase=False)
    np.testing.assert_almost_equal(result, 46.51, decimal=2)


def test_get_moses_multi_bleu_lowercase():
    hypotheses = ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"]
    references = [
        "The quick brown fox jumps over the lazy dog 笑",
        "The quick brown fox jumps over the lazy dog 笑"
    ]
    result = get_moses_multi_bleu(hypotheses, references, lowercase=True)
    np.testing.assert_almost_equal(result, 46.51, decimal=2)
