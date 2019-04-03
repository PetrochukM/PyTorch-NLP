from itertools import product


def kwargs_product(dict_):
    """
    Args:
        dict_ (dict): Dict with possible kwargs values.

    Returns:
        (iterable) Iterable over all combinations of the dict of kwargs.

    Usage:

        >>> list(kwargs_product({ 'number': [1,2], 'character': 'ab' }))
        [{'number': 1, 'character': 'a'}, {'number': 1, 'character': 'b'}, \
{'number': 2, 'character': 'a'}, {'number': 2, 'character': 'b'}]

    """
    return (dict(zip(dict_, x)) for x in product(*dict_.values()))
