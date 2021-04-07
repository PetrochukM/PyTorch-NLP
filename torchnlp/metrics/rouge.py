import itertools
import numpy as np

def _get_ngrams(n, text):
  
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set

def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = split_into_words(sentences)
  return _get_ngrams(n, words)

def rouge_n(evaluated_sentences, reference_sentences, n=2):
  
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score 

def len_lcs(x, y):
  
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table[n,m]


def split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(sentences.split(" "))


def rogue_l( candidate, references  ):
  lcs = len_lcs ( candidate, references )
  len_x = len(candidate)
  len_y = len(references)

  recall = lcs / len_y
  precision = lcs / len_x 
  beta = precision/ (recall + 1e-12)
  numerator = (1 + (beta ** 2 ) )* (  precision * recall )
  denominator = ( precision* ( beta ** 2 ) + recall  ) + 1e-8
  f1_score = numerator/ denominator 
  return f1_score

def average_rouge ( candidate, references ):
  rouge_1 = rouge_n( candidate, references, 1 )
  rouge_2 = rouge_n( candidate, references, 2 )
  rouge_lcs = rogue_l( split_into_words(candidate), split_into_words(references) )
  avg_rouge = (rouge_1+rouge_2+rouge_lcs)/3
  print("rouge_1:", rouge_1)
  print("rouge_2:", rouge_2)
  print("rouge_lcs:", rouge_lcs)
  print("average:" ,avg_rouge)
    
def main():
    x = "The quick brown fox jumped over the wall"
    y = "The fast black dog and fox jumped into the wall"
    x_words = split_into_words(x)
    y_words = split_into_words(y)
    print(x_words)
    lcs = len_lcs(x_words,y_words)
    average_rouge(x, y )
    
if __main__ == "main":
    main()
