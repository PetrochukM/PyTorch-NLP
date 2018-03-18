from torchnlp.samplers import BPTTBatchSampler


def test_bptt_batch_sampler_drop_last():
    # Test samplers iterate over chunks similar to:
    # https://github.com/pytorch/examples/blob/c66593f1699ece14a4a2f4d314f1afb03c6793d9/word_language_model/main.py#L112
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    sampler = BPTTBatchSampler(alphabet, bptt_length=2, batch_size=4, drop_last=True)
    decoded_batches = []
    for batch in list(sampler):
        decoded_batch = []
        for source, target in batch:
            decoded_batch.append([alphabet[source], alphabet[target]])
        decoded_batches.append(decoded_batch)
    assert decoded_batches[0] == [[['a', 'b'], ['b', 'c']], [['g', 'h'], ['h', 'i']],
                                  [['m', 'n'], ['n', 'o']], [['s', 't'], ['t', 'u']]]

    assert len(sampler) == len(decoded_batches)


def test_bptt_batch_sampler():
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    sampler = BPTTBatchSampler(alphabet, bptt_length=2, batch_size=4, drop_last=False)
    decoded_batches = []
    for batch in list(sampler):
        decoded_batch = []
        for source, target in batch:
            decoded_batch.append([alphabet[source], alphabet[target]])
        decoded_batches.append(decoded_batch)
    assert decoded_batches[0] == [[['a', 'b'], ['b', 'c']], [['h', 'i'], ['i', 'j']],
                                  [['o', 'p'], ['p', 'q']], [['u', 'v'], ['v', 'w']]]
    assert len(sampler) == len(decoded_batches)


def test_bptt_batch_sampler_example():
    sampler = BPTTBatchSampler(range(100), bptt_length=2, batch_size=3, drop_last=False)
    assert list(sampler)[0] == [(slice(0, 2), slice(1, 3)), (slice(34, 36), slice(35, 37)), (slice(
        67, 69), slice(68, 70))]
    assert list(sampler)[1] == [(slice(2, 4), slice(3, 5)), (slice(36, 38), slice(37, 39)), (slice(
        69, 71), slice(70, 72))]
