import pickle

from torchnlp.samplers import RepeatSampler


def test_repeat_sampler():
    sampler = RepeatSampler([1])
    iterator = iter(sampler)
    assert next(iterator) == 1
    assert next(iterator) == 1
    assert next(iterator) == 1


def test_pickleable():
    sampler = RepeatSampler([1])
    pickle.dumps(sampler)
