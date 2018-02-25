import logging

import pandas as pd
import random

from lib.utils import torch_equals_ignore_index

logger = logging.getLogger(__name__)

# TODO: Why require decoding? The original source can just be sent? Maybe overall the thing is incorreect.


def print_random_sample(sources,
                        targets,
                        outputs,
                        input_text_encoder,
                        output_text_encoder,
                        n_samples=5,
                        ignore_index=None):
    """ Print a random sample of positive and negative samples """
    positive_indexes = []
    negative_indexes = []
    predictions = []
    for i, (target, output) in enumerate(zip(targets, outputs)):
        target = target.squeeze(dim=0)
        output = output.squeeze(dim=0)
        prediction = output.max(output.dim() - 1)[1].view(-1)
        predictions.append(prediction)
        if torch_equals_ignore_index(target, prediction, ignore_index=ignore_index):
            positive_indexes.append(i)
        else:
            negative_indexes.append(i)
    positive_samples = random.sample(positive_indexes, min(len(positive_indexes), n_samples))
    negative_samples = random.sample(negative_indexes, min(len(negative_indexes), n_samples))

    ret = 'Random Sample:\n'
    for prefix, samples in [('Positive', positive_samples), ('Negative', negative_samples)]:
        data = []
        for i in samples:
            source = input_text_encoder.decode(sources[i].squeeze(dim=0))
            target = output_text_encoder.decode(targets[i].squeeze(dim=0))
            prediction = output_text_encoder.decode(predictions[i])
            data.append([source, target, prediction])
        ret += '\n%s Samples:\n%s\n' % (
            prefix, pd.DataFrame(data, columns=['Source', 'Target', 'Prediction']))

    logger.info(ret)
