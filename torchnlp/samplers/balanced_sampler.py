from torchnlp._third_party.weighted_random_sampler import WeightedRandomSampler

from torchnlp.utils import identity


class BalancedSampler(WeightedRandomSampler):
    """ Weighted sampler with respect for an element's class.

    Args:
        data (iterable)
        get_class (callable, optional): Get the class of an item relative to the entire dataset.
        get_weight (callable, optional): Define a weight for each item other than one.
        kwargs: Additional key word arguments passed onto `WeightedRandomSampler`.

    Example:
        >>> from torchnlp.samplers import DeterministicSampler
        >>>
        >>> data = ['a', 'b', 'c'] + ['c'] * 100
        >>> sampler = BalancedSampler(data, num_samples=3)
        >>> sampler = DeterministicSampler(sampler, random_seed=12)
        >>> [data[i] for i in sampler]
        ['c', 'b', 'a']
    """

    def __init__(self, data_source, get_class=identity, get_weight=lambda x: 1, **kwargs):
        classified = [get_class(item) for item in data_source]
        weighted = [float(get_weight(item)) for item in data_source]
        class_totals = {
            k: sum([w for c, w in zip(classified, weighted) if k == c]) for k in set(classified)
        }
        weights = [w / class_totals[c] if w > 0 else 0.0 for c, w in zip(classified, weighted)]
        super().__init__(weights=weights, **kwargs)
