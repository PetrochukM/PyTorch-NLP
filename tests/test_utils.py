import os
import sys

import torch

from torchnlp.datasets import Dataset
from torchnlp.text_encoders import PADDING_INDEX
from torchnlp.utils import batch
from torchnlp.utils import cuda_devices
from torchnlp.utils import device_default
from torchnlp.utils import get_root_path
from torchnlp.utils import get_total_parameters
from torchnlp.utils import new_experiment_folder
from torchnlp.utils import pad_batch
from torchnlp.utils import pad_tensor
from torchnlp.utils import resplit_datasets
from torchnlp.utils import seed
from torchnlp.utils import torch_equals_ignore_index
from torchnlp.utils import save_standard_streams
from tests.utils import MockModel


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
    a, b = resplit_datasets(a, b, random_seed=123, cut=0.3)
    assert len(a) == 3
    assert len(b) == 7


def test_get_root_path():
    root_path = get_root_path()
    assert os.path.isfile(os.path.join(root_path, 'requirements.txt'))


def test_new_experiment_folder():
    directory = 'tests/_test_data/experiments'
    path = new_experiment_folder(parent_directory=directory)
    assert os.path.isdir(path)
    os.rmdir(path)
    os.rmdir(directory)


def test_batch_generator():

    def generator():
        for i in range(11):
            yield i

    assert len(list(batch(generator(), n=2))) == 6


def test_batch():
    assert len(list(batch([i for i in range(11)], n=2))) == 6


def test_device_default():
    """ Check device default
    """
    assert device_default(1) == 1
    assert device_default(-1) == -1
    if torch.cuda.is_available():
        assert device_default(None) is None
    else:
        assert device_default(None) == -1


def test_cuda_devices():
    """ Run CUDA devices
    """
    if torch.cuda.is_available():
        cuda_devices()
    else:
        assert cuda_devices() == []


def test_get_total_parameters():
    model = MockModel()
    assert 62006 == get_total_parameters(model)


def test_pad_tensor():
    padded = pad_tensor(torch.LongTensor([1, 2, 3]), 5)
    assert padded.tolist() == [1, 2, 3, PADDING_INDEX, PADDING_INDEX]


def test_pad_batch():
    batch = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2]), torch.LongTensor([1])]
    padded, lengths = pad_batch(batch)
    padded = [r.tolist() for r in padded]
    assert padded == [[1, 2, 3], [1, 2, PADDING_INDEX], [1, PADDING_INDEX, PADDING_INDEX]]
    assert lengths == [3, 2, 1]


def test_seed():
    # Just make sure it works
    seed(123)


def test_torch_equals_ignore_index():
    source = torch.LongTensor([1, 2, 3])
    target = torch.LongTensor([1, 2, 4])
    assert torch_equals_ignore_index(source, target, ignore_index=3)
    assert not torch_equals_ignore_index(source, target)


def test_save_standard_streams():
    directory = os.path.dirname(os.path.realpath(__file__))
    save_standard_streams(directory, 'stdout.log', 'stderr.log')

    # Check if 'Test' gets captured
    print('Test')

    # Reset stdout and stderr streams
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = sys.stdout.stream
    sys.stderr = sys.stderr.stream

    stdout = os.path.join(directory, 'stdout.log')
    stderr = os.path.join(directory, 'stderr.log')
    assert os.path.isfile(stdout)
    assert os.path.isfile(stderr)

    # Just `Test` print in stdout
    lines = [l.strip() for l in open(stdout, 'r')]
    assert lines[0] == 'Test'

    # Nothing in stderr
    lines = [l.strip() for l in open(stderr, 'r')]
    assert len(lines) == 0

    # Clean up files
    os.remove(stdout)
    os.remove(stderr)
