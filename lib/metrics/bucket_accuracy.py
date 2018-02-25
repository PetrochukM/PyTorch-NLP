import logging

import pandas as pd

from lib.metrics.accuracy import get_accuracy

logger = logging.getLogger(__name__)


def print_bucket_accuracy(buckets, targets, outputs, ignore_index=None):
    """
    Args:
      buckets (list of keys)
      targets (list of tensors)
      outputs (list of tensors)
      ignore_index (int, optional): specifies a target index that is ignored
    """
    keys = list(set(buckets))
    bucketed_targets = {key: [] for key in keys}
    bucketed_outputs = {key: [] for key in keys}
    for key, target, output in zip(buckets, targets, outputs):
        bucketed_targets[key].append(target)
        bucketed_outputs[key].append(output)

    columns = ['Accuracy', 'Num Correct', 'Num Total']
    data = []
    for key in keys:
        accuracy, n_correct, n_total = get_accuracy(
            bucketed_targets[key], bucketed_outputs[key], ignore_index=ignore_index)
        data.append([accuracy, n_correct, n_total])

    df = pd.DataFrame(data, index=keys, columns=columns)
    logger.info('Bucket Accuracy:\n%s\n', df)
