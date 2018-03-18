import urllib.request
import pickle

import torch

from tqdm import tqdm

from torchnlp.datasets import Dataset
from torchnlp.utils import flatten_parameters
from torchnlp.utils import pad_batch
from torchnlp.utils import pad_tensor
from torchnlp.utils import reporthook
from torchnlp.utils import resplit_datasets
from torchnlp.utils import shuffle
from torchnlp.utils import torch_equals_ignore_index


def test_pad_tensor():
    PADDING_INDEX = 0
    padded = pad_tensor(torch.LongTensor([1, 2, 3]), 5, PADDING_INDEX)
    assert padded.tolist() == [1, 2, 3, PADDING_INDEX, PADDING_INDEX]


def test_pad_batch():
    PADDING_INDEX = 0
    batch = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2]), torch.LongTensor([1])]
    padded, lengths = pad_batch(batch, PADDING_INDEX)
    padded = [r.tolist() for r in padded]
    assert padded == [[1, 2, 3], [1, 2, PADDING_INDEX], [1, PADDING_INDEX, PADDING_INDEX]]
    assert lengths == [3, 2, 1]


def test_shuffle():
    a = [1, 2, 3, 4, 5]
    # Always shuffles the same way
    shuffle(a)
    assert a == [4, 2, 5, 3, 1]


def test_flatten_parameters():
    rnn = torch.nn.LSTM(10, 20, 2)
    rnn_pickle = pickle.dumps(rnn)
    rnn2 = pickle.loads(rnn_pickle)
    # Check that ``flatten_parameters`` works with a RNN module.
    flatten_parameters(rnn2)


def test_reporthook():
    # Check that reporthook works with URLLIB
    with tqdm(unit='B', unit_scale=True, miniters=1) as t:
        urllib.request.urlretrieve('http://google.com', reporthook=reporthook(t))


def test_resplit_datasets():
    a = Dataset([{'r': 1}, {'r': 2}, {'r': 3}, {'r': 4}, {'r': 5}])
    b = Dataset([{'r': 6}, {'r': 7}, {'r': 8}, {'r': 9}, {'r': 10}])
    # Test determinism
    a, b = resplit_datasets(a, b, random_seed=123)
    assert list(a) == [{'r': 9}, {'r': 8}, {'r': 6}, {'r': 10}, {'r': 3}]
    assert list(b) == [{'r': 4}, {'r': 7}, {'r': 2}, {'r': 5}, {'r': 1}]


def test_resplit_datasets_cut():
    a = Dataset([{'r': 1}, {'r': 2}, {'r': 3}, {'r': 4}, {'r': 5}])
    b = Dataset([{'r': 6}, {'r': 7}, {'r': 8}, {'r': 9}, {'r': 10}])
    a, b = resplit_datasets(a, b, random_seed=123, split=0.3)
    assert len(a) == 3
    assert len(b) == 7


def test_torch_equals_ignore_index():
    source = torch.LongTensor([1, 2, 3])
    target = torch.LongTensor([1, 2, 4])
    assert torch_equals_ignore_index(source, target, ignore_index=3)
    assert not torch_equals_ignore_index(source, target)
