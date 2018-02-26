from itertools import product

import torch
from torch.autograd import Variable


def kwargs_product(dict_):
    """
    Args:
        dict_ (dict): dict with possible kwargs values
    Returns:
        (iterable) iterable over all combinations of the dict of kwargs
    Usage:
        >>> list(dict_product(dict(number=[1,2], character='ab')))
        [{'character': 'a', 'number': 1},
        {'character': 'a', 'number': 2},
        {'character': 'b', 'number': 1},
        {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dict_, x)) for x in product(*dict_.values()))


def tensor(*args, type_=torch.LongTensor, max_=100, variable=True):
    """
    Args:
        type_ constructor for a tensor
    Returns:
        type_ [*args] filled with random numbers from a uniform distribution [0, max]
    """
    ret = type_(*args).random_(to=max_ - 1)
    if variable:
        ret = Variable(ret)
    return ret
