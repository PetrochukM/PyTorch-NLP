import unittest

import torch
from torch.optim.lr_scheduler import StepLR
import mock

from lib.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            optimizer = Optimizer(torch.optim.Adam(params))
        except:
            self.fail("__init__ failed.")

        self.assertEquals(optimizer.max_grad_norm, 0)

    def test_update(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        optimizer = Optimizer(torch.optim.Adam(params, lr=1), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1, gamma=0.1)
        optimizer.set_scheduler(scheduler)
        optimizer.update(10, 0)
        optimizer.update(10, 1)
        self.assertEquals(optimizer.optimizer.param_groups[0]['lr'], 0.1)

    @mock.patch("torch.nn.utils.clip_grad_norm")
    def test_step(self, mock_clip_grad_norm):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        optim = Optimizer(torch.optim.Adam(params), max_grad_norm=5)
        optim.step()
        mock_clip_grad_norm.assert_called_once()
