import numpy as np

class Ngrams(object):
    """
    datastructure for n grams.
    if `exclusive`, datastructure is set
    otherwise, datastructure is list
    """
    def __init__(self, ngrams={}, exclusive=True):
        self.exclusive = exclusive
        if exclusive:
            self._grams = set(ngrams)
        else:
            self._grams = list(ngrams)
    def __len__(self):
        return len(self._grams)
    def add(self, elem):
        if self.exclusive:
            self._grams.add(elem)
        else:
            self._grams.append(elem)
    def intersection(self, other_gram):
        if self.exclusive:
            inter_set = self._grams.intersection(other_gram._grams)
            return Ngrams(inter_set, exclusive=True)
        else:
            other_dict = dict()
            inter_list = list()
            for gram in other_gram._grams:
                other_dict[gram] = other_dict.get(gram, 0) + 1
            for gram in self._grams:
                if gram in other_dict and other_dict[gram] > 0:
                    other_dict[gram] -= 1
                    inter_list.append(gram)
            return Ngrams(inter_list, exclusive=False)


def _get_ngrams(n, text, exclusive):
    """
    calculate the n-grams.
    Args:
        n: n-gram to calculate
        text: An array of tokens
        exclusive: if True, the datastructure is set, else is set.
    Returns:
        A set of n-grams
    """
    ngram_set = Ngrams(exclusive=exclusive)
    if type(text) == str:
        text = text.split()
    text_length = len(text)
    index_ngram_end = text_length - n
    for i in range(index_ngram_end + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _frp_rouge_n(eval_count, ref_count, overlapping_count):
    """
    compute f score, precision socre and recall score.
    Args:
        eval_count: the evaluation sentence n-gram count
        ref_count: the reference sentence n-gram count
        overlapping_count: the overlapping n-gram between evaluation and reference. 
    """
    if eval_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / eval_count

    if ref_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / ref_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return {"f": f1_score, "p": precision, "r": recall}


def get_rouge_n(evaluated_sentences, reference_sentences, n=2, exclusive=True):
    """
    Computes ROUGE-N of two text collections of sentences, namely evaluated_sentences and
        reference senteces
    Args:
        evaluated_sentences: The sentences that have been produced by the
                           summarizer
        reference_sentences: The sentences from the referene set
        n: Size of ngram.  Defaults to 2.
    Returns:
        tuple. (f1, precision, recall) for ROUGE-N
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Hypothesis set is empty.")
    if len(reference_sentences) <= 0:
        raise ValueError("reference set is empty")
    for eval_sentence, ref_sentence in zip(evaluated_sentences, reference_sentences):
        eval_ngrams = _get_ngrams(n, eval_sentence, exclusive)
        ref_grams = _get_ngrams(n, ref_sentence, exclusive)
        ref_count = len(ref_grams)
        eval_count = len(eval_ngrams)
        overlapping_ngrams = eval_ngrams.intersection(ref_grams)
        overlapping_count = len(overlapping_ngrams)
    return _frp_rouge_n(eval_count, ref_count, overlapping_count)
    