import itertools
import math
import random

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
    compute f score, precision score and recall score.
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


def _frp_rouge_l(llcs, m, n, beta=None):
    """
    Computes the LCS-based F-measure score, Precision score and recall score.
    Args:
        llcs: Length of LCS
        m: number of words in reference summary
        n: number of words in candidate summary
        beta: beta = P_lcs / R_lcs when ∂ F_lcs / ∂ R_lcs = ∂ F_lcs / ∂ P_lcs. In DUC, beta is set to a very big number, and only R_lcs is considered.
    Returns:
        dictionary. 'f' for F-measure score, 'p' for Precision score and 'r' for recall score.
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    if not beta:
        beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return {"f": f_lcs, "p": p_lcs, "r": r_lcs}


def check_inverse(f, inv_f, eps=1e-5):
    """
    Check if function inv_f is the inverse function of function f
    Args:
        f: a function
        inv_f: inverse function of f
        eps: error threshold
    Returns:
        bool, if inv_f is the inverse function of f
    """
    for i in range(100000):
        x = random.random() * 100000
        y = f(x)
        _x = inv_f(y)
        if abs(x-_x) > eps:
            return False
    return True


def check_increase(f):
    """
    Check if function f satisfies f(x + y) > f(x) + f(y)
    Args:
        f: function
    Returns:
        bool, if f(x + y) > f(x) + f(y) or not
    """
    for i in range(100000):
        x = random.random() * 100000
        y = random.random() * 100000
        if f(x)+f(y) <= f(x+y):
            return False
    return True


def _frp_rouge_w(wlcs, m, n, f=lambda x: x**2, inv_f=lambda x: math.sqrt(x), beta=None, strict=False):
    """
    Computes the LCS-based F-measure score.
    Args:
       wlcs: wlcs score
       m: number of words in reference summary
       n: number of words in candidate summary
       f: weighting function
       inv_f: inverse function of weighting function
       beta: beta = P_lcs / R_lcs when ∂ F_lcs / ∂ R_lcs = ∂ F_lcs / ∂ P_lcs. In DUC, beta is set to a very big number, and only R_lcs is considered.
    Returns:
       dictionary. WLCS-based F-measure score, P-score and R-score
    """
    if strict:
        assert(check_increase(f))
        assert(check_inverse(f, inv_f))
    r_wlcs = inv_f(wlcs/f(m))
    p_wlcs = inv_f(wlcs/f(n))
    if not beta:
        beta = p_wlcs / (r_wlcs + 1e-12)
    num = (1 + (beta**2)) * r_wlcs * p_wlcs
    denom = r_wlcs + ((beta**2) * p_wlcs)
    f_wlcs = num / (denom + 1e-12)
    return {"f": f_wlcs, "p": p_wlcs, "r": r_wlcs}


def lcs(x, y):
    """
    Compute the length of longest common sequence of x and y
    Args:
        x, y: List of element
    Returns:
        Dictionary, table[i, j] represent the length of longest common sequence of x[0: i-1] and y[0: j-1]
    """
    n, m = len(x), len(y)
    table = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def lcs_seq(x, y):
    """
    Find the longest common sequence of x and y
    Args:
        x, y: List of element
    Returns:
        List, the longest common sequence of x and y
    """
    table = lcs(x, y)

    def lcs_seq_wrd(i, j):
        if i == 0 or j == 0:
            return []
        if x[i-1] == y[j-1]:
            return lcs_seq_wrd(i-1, j-1) + [x[i-1]]
        elif table[i-1, j] > table[i, j-1]:
            return lcs_seq_wrd(i-1, j)
        else:
            return lcs_seq_wrd(i, j-1)
    return lcs_seq_wrd(len(x), len(y))


def _w_lcs(x, y, func=lambda x: x**2):
    """
    Compute the we of weighted longest common sequence of x and y,

    c is the dynamic programming table, c(i,j) stores the WLCS score ending at word x[i] of X and y[j] of Y.
    w is the table storing the length of consecutive matches ended at c table position i and j, and f is a function of consecutive matches at the table position, c(i, j).

    Args:
        x, y: List of element
        func: the weighting function which should satisfies f(x+y) > f(x) + f(y) for any positive integers x and y, and should hava a close form inverse function.
    Returns:
        Float, the WLCS score of x and y
    """
    n = len(x)
    m = len(y)
    c = {}
    w = {}
    for i in range(0, n + 1):
        c[i, 0] = 0
        w[i, 0] = 0
    for j in range(0, m + 1):
        c[0, j] = 0
        w[0, j] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                k = w[i - 1, j - 1]
                c[i, j] = c[i - 1, j - 1] + func(k + 1) - func(k)
                w[i, j] = k + 1
            else:
                c[i, j] = max(c[i - 1, j], c[i, j - 1])
                w[i, j] = 0
    return c[n, m]


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
        raise ValueError("reference set is empty.")
    for eval_sentence, ref_sentence in zip(evaluated_sentences, reference_sentences):
        eval_ngrams = _get_ngrams(n, eval_sentence, exclusive)
        ref_grams = _get_ngrams(n, ref_sentence, exclusive)
        ref_count = len(ref_grams)
        eval_count = len(eval_ngrams)
        overlapping_ngrams = eval_ngrams.intersection(ref_grams)
        overlapping_count = len(overlapping_ngrams)
    return _frp_rouge_n(eval_count, ref_count, overlapping_count)


def get_rouge_l_summary_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L of two summary-level sentences, namely evaluated_sentences and reference senteces
    Reference: https://www.aclweb.org/anthology/W04-1013.pdf
    Args:
        evaluated_sentences: List of sentences that have been produced by the summarizer
        reference_sentences: List of sentences from the reference set
    Returns:
        dictionary. {'f': f1_score, 'p': precision, 'r': recall} for ROUGE-L
    """
    if len(evaluated_sentences) <= 0:
        raise ValueError("Hypothesis set is empty.")
    if len(reference_sentences) <= 0:
        raise ValueError("reference set is empty")

    llcs = 0
    ref_words = [w for s in reference_sentences for w in s.split()]
    eval_words = [w for s in evaluated_sentences for w in s.split()]
    m = len(set(ref_words))
    n = len(set(eval_words))

    reference_sentences = [s.split() for s in reference_sentences]
    evaluated_sentences = [s.split() for s in evaluated_sentences]
    for evaluated_sentence in evaluated_sentences:
        union_lcs = set()
        for reference_sentence in reference_sentences:
            union_lcs |= set(lcs_seq(reference_sentence, evaluated_sentence))
        llcs += len(union_lcs)
    return _frp_rouge_l(llcs, m, n)


def get_rouge_w(evaluated_sentence, reference_sentence, f=lambda x: x**2, inv_f=lambda x: math.sqrt(x)):
    """
    Computes ROUGE-W of two sequences, namely evaluated_sentence and reference sentece
    Reference: https://www.aclweb.org/anthology/W04-1013.pdf
    Args:
        evaluated_sentence: a sentence that have been produced by the summarizer
        reference_sentence: a sentence from the reference set
    Returns:
        dictionary. {'f': f1_score, 'p': precision, 'r': recall} for ROUGE-W
    """
    if not type(evaluated_sentence) == str:
        raise ValueError("Hypothesis should be a sentence.")
    if not type(reference_sentence) == str:
        raise ValueError("reference should be a sentence")
    eval_sentence = evaluated_sentence.split()
    ref_sentence = reference_sentence.split()
    n = len(ref_sentence)
    m = len(eval_sentence)
    wlcs = _w_lcs(eval_sentence, ref_sentence)
    return _frp_rouge_w(wlcs, n, m, f, inv_f)
