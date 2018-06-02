from itertools import product


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
