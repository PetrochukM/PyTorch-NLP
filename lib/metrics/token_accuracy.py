import logging

logger = logging.getLogger(__name__)


def get_token_accuracy(targets, outputs, ignore_index=None, print_=False):
    """ Compute the token accuracy. """
    n_correct = 0
    n_total = 0
    for target, output in zip(targets, outputs):
        target = target.squeeze(dim=0)
        output = output.squeeze(dim=0)
        prediction = output.max(output.dim() - 1)[1].view(-1)
        if ignore_index is not None:
            mask = target.ne(ignore_index)
            n_correct += prediction.eq(target).masked_select(mask).sum()
            n_total += mask.sum()
        else:
            n_total += len(target)
            n_correct += prediction.eq(target).sum()
    token_accuracy = float(n_correct) / n_total
    if print_:
        logger.info('Token Accuracy: %s [%d of %d]', token_accuracy, n_correct, n_total)
    return token_accuracy, n_correct, n_total
