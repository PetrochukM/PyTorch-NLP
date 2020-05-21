from torchnlp.metrics.accuracy import get_accuracy, get_token_accuracy
from torchnlp.metrics.bleu import get_moses_multi_bleu
from torchnlp.metrics.rouge import get_rouge_l_summary_level, get_rouge_n, get_rouge_w

# TODO: Use `sklearn.metrics` for a `confusion_matrix` implemented with ignore_index
# TODO: Use `sklearn.metrics` for a `recall` implemented with ignore_index
# TODO: Use `sklearn.metrics` for a `precision` implemented with ignore_index
# TODO: Use `sklearn.metrics` for a `f1` implemented with ignore_index
# TODO: Implement perplexity
# TODO: Implement rogue metric

__all__ = ['get_accuracy', 'get_token_accuracy', 'get_moses_multi_bleu', 'get_rouge_n', 'get_rouge_l_summary_level', 'get_rouge_w']
