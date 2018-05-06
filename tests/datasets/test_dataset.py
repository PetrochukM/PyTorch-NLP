import pytest
import random

from torchnlp.datasets import Dataset


def test_dataset_init():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    assert len(dataset) == 2
    assert 'a' in dataset
    assert 'b' in dataset
    assert 'c' not in dataset


def test_dataset_str():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    assert '    a   b\n0   a   b\n1  aa  bb' == str(dataset)


def test_dataset_get_column():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    assert dataset['a'] == ['a', 'aa']
    assert dataset['b'] == ['b', 'bb']
    with pytest.raises(AttributeError):
        dataset['c']


def test_dataset_set_column():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])

    # Regular column update
    dataset['a'] = ['aa', 'aaa']
    assert dataset['a'] == ['aa', 'aaa']

    # To Little
    dataset['b'] = ['b']
    assert dataset['b'] == ['b', None]

    # Too many
    dataset['c'] = ['c', 'cc', 'ccc']
    assert dataset['c'] == ['c', 'cc', 'ccc']

    # Smoke (regression test)
    random.shuffle(dataset)


def test_dataset_get_row():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    assert dataset[0] == {'a': 'a', 'b': 'b'}
    assert dataset[1] == {'a': 'aa', 'b': 'bb'}
    with pytest.raises(IndexError):
        dataset[2]


def test_dataset_set_row():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    dataset[0] = {'c': 'c'}
    assert dataset['c'] == ['c', None]
    assert dataset['a'] == [None, 'aa']

    dataset[0:2] = [{'d': 'd'}, {'d': 'dd'}]
    assert dataset[0] == {'d': 'd'}
    assert dataset[1] == {'d': 'dd'}

    with pytest.raises(IndexError):
        dataset[2] = {'c': 'c'}


def test_dataset_equality():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    other_dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    assert dataset == other_dataset


def test_dataset_concat():
    dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    other_dataset = Dataset([{'a': 'a', 'b': 'b'}, {'a': 'aa', 'b': 'bb'}])
    concat = dataset + other_dataset
    assert len(concat) == 4
    assert list(concat) == dataset.rows + other_dataset.rows
