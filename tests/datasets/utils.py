def try_dataset(func):
    """
    Generic function to test a dataset loader.
    """
    test_data = func(test=True)
    train_data, dev_data, test_data = func(test=True, train=True, dev=True)
    assert list(test_data) == list(test_data)
    assert len(train_data) > 0
    assert len(dev_data) > 0
    assert len(test_data) > 0
