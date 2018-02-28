from torchnlp.samplers import BucketBatchSampler


def test_bucket_batch_sampler():
    data_source = [[1], [2], [3], [4], [5], [6]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(BucketBatchSampler(data_source, sort_key, batch_size))
    assert len(batches) == 3


def test_bucket_batch_sampler_uneven():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(BucketBatchSampler(data_source, sort_key, batch_size))
    assert len(batches) == 3
    batches = list(BucketBatchSampler(data_source, sort_key, batch_size, drop_last=True))
    assert len(batches) == 2


def test_bucket_batch_sampler_last_batch_first():
    data_source = [[1], [2], [3], [4], [5, 6, 7, 8, 9, 10]]
    sort_key = lambda r: len(r)
    batch_size = 2
    batches = list(BucketBatchSampler(data_source, sort_key, batch_size, last_batch_first=True))
    # Largest batch (4) is in first batch
    assert 4 in batches[0]


def test_bucket_batch_sampler_sorted():
    data_source = [[1], [2], [3], [4], [5]]
    sort_key = lambda r: r[0]
    batch_size = 1
    batches = list(
        BucketBatchSampler(
            data_source,
            sort_key,
            batch_size,
            shuffle=False,
            last_batch_first=False,
            sort_key_noise=0.0))
    # Largest batch (4) is in first batch
    for i, batch in enumerate(batches):
        assert batch[0] == i
