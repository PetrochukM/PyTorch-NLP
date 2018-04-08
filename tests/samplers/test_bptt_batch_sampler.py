from torchnlp.samplers import BPTTBatchSampler
from torchnlp.utils import sampler_to_iterator


def test_bptt_batch_sampler_drop_last():
    # Test samplers iterate over chunks similar to:
    # https://github.com/pytorch/examples/blob/c66593f1699ece14a4a2f4d314f1afb03c6793d9/word_language_model/main.py#L112
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    sampler = BPTTBatchSampler(alphabet, bptt_length=2, batch_size=4, drop_last=True)
    list_ = list(sampler_to_iterator(alphabet, sampler))
    assert list_[0] == [['a', 'b'], ['g', 'h'], ['m', 'n'], ['s', 't']]
    assert len(sampler) == len(list_)


def test_bptt_batch_sampler():
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    sampler = BPTTBatchSampler(alphabet, bptt_length=2, batch_size=4, drop_last=False)
    list_ = list(sampler_to_iterator(alphabet, sampler))
    assert list_[0] == [['a', 'b'], ['h', 'i'], ['o', 'p'], ['u', 'v']]
    assert len(sampler) == len(list_)


def test_bptt_batch_sampler_example():
    sampler = BPTTBatchSampler(range(100), bptt_length=2, batch_size=3, drop_last=False)
    assert list(sampler)[0] == [slice(0, 2), slice(34, 36), slice(67, 69)]

    sampler = BPTTBatchSampler(
        range(100), bptt_length=2, batch_size=3, drop_last=False, type_='target')
    assert list(sampler)[0] == [slice(1, 3), slice(35, 37), slice(68, 70)]
