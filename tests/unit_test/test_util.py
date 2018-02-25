import unittest
import os
import shutil

import torch

from lib.utils import _init_logging_return
from lib.utils import cuda_devices
from lib.utils import device_default
from lib.utils import get_log_directory_path
from lib.utils import get_root_path
from lib.utils import get_total_parameters
from lib.utils import init_logging
from lib.utils import setup_training
from lib.utils import pad_tensor
from lib.utils import pad_batch
from lib.utils import torch_equals_ignore_index
from lib.text_encoders import PADDING_INDEX
from tests.lib.utils import random_args


class TestUtil(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Set up some useful arguments in `self`
        for key, value in random_args(False).items():
            setattr(self, key, value)

    def test_get_root_path(self):
        """ Check the root path is right.
        """
        root_path = get_root_path()
        assert os.path.isdir(root_path)
        src_dir = os.path.join(root_path, 'lib')
        assert os.path.isdir(src_dir)

    def test_init_logging(self):
        """ Test if logging configuration was set.
        """
        global _init_logging_return
        # Set it up
        log_directory = 'logs/test/'
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        set_log_directory = init_logging(log_directory)
        self.assertEqual(set_log_directory, log_directory)

        # Secondary log directory does not work
        log_directory = 'logs_2/tests/'
        set_log_directory = init_logging(log_directory)
        self.assertNotEqual(set_log_directory, log_directory)

        _init_logging_return = None
        shutil.rmtree(set_log_directory)

    def test_get_log_directory_path(self):
        log_directory = get_log_directory_path('test')
        self.assertTrue(os.path.isdir(log_directory))
        self.assertTrue(os.path.exists(log_directory))
        os.rmdir(log_directory)

    def test_cuda_devices(self):
        """ Run CUDA devices
        """
        if torch.cuda.is_available():
            self.assertTrue(len(cuda_devices()) > 0)
        else:
            self.assertEqual(cuda_devices(), [])

    def test_device_default(self):
        """ Check device default
        """
        self.assertEqual(device_default(1), 1)
        self.assertEqual(device_default(-1), -1)
        if torch.cuda.is_available():
            self.assertEqual(device_default(None), torch.cuda.current_device())
        else:
            self.assertEqual(device_default(None), -1)

    def test_get_total_parameters(self):
        n_params = get_total_parameters(self.model)
        self.assertTrue(n_params > 0)

    def test_pad_tensor(self):
        tensor = torch.LongTensor([4])
        padded = pad_tensor(tensor, 3)
        padded_expected = torch.LongTensor([4, PADDING_INDEX, PADDING_INDEX])
        self.assertTrue(torch.equal(padded, padded_expected))

    def test_pad_batch(self):
        batch = [torch.LongTensor([1]), torch.LongTensor([1, 2]), torch.LongTensor([1, 2, 3])]
        padded, padded_lengths = pad_batch(batch)
        self.assertEqual(len(padded), len(batch))
        self.assertTrue(torch.equal(padded[0], torch.LongTensor([1, PADDING_INDEX, PADDING_INDEX])))
        self.assertTrue(torch.equal(padded[1], torch.LongTensor([1, 2, PADDING_INDEX])))
        self.assertTrue(torch.equal(padded[2], torch.LongTensor([1, 2, 3])))
        self.assertEqual(padded_lengths, [1, 2, 3])

    def test_setup_training(self):
        checkpoint_path = ''
        log_directory = 'logs/'
        device = -1
        random_seed = 123
        # Make sure nothing dies
        is_cuda, checkpoint = setup_training(checkpoint_path, log_directory, device, random_seed)
        self.assertIsNone(checkpoint)

    def test_torch_equals_ignore_index(self):
        self.assertTrue(
            torch_equals_ignore_index(
                torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2, 4]), ignore_index=3))
        self.assertFalse(
            torch_equals_ignore_index(
                torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2, 4]), ignore_index=4))
