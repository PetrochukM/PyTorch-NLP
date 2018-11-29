import torch
import pytest

from torchnlp.nn.weight_drop import WeightDropLSTM
from torchnlp.nn.weight_drop import WeightDropGRU
from torchnlp.nn.weight_drop import WeightDropLinear
from torchnlp.nn.weight_drop import WeightDrop


def test_weight_drop_linear():
    # Input is (seq, batch, input)
    x = torch.randn(2, 1, 10)

    lin = WeightDropLinear(10, 10, weight_dropout=0.9)
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]


def test_weight_drop_lstm():
    input_ = torch.randn(2, 1, 10)

    wd_lstm = WeightDropLSTM(10, 10, num_layers=2, weight_dropout=0.9)
    run1 = [x.sum() for x in wd_lstm(input_)[0].data]
    run2 = [x.sum() for x in wd_lstm(input_)[0].data]

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert pytest.approx(run1[0].item()) == pytest.approx(run2[0].item())
    # Second step should not
    assert run1[1] != run2[1]


def test_weight_drop_gru():
    input_ = torch.randn(2, 1, 10)

    wd_lstm = WeightDropGRU(10, 10, num_layers=2, weight_dropout=0.9)
    run1 = [x.sum() for x in wd_lstm(input_)[0].data]
    run2 = [x.sum() for x in wd_lstm(input_)[0].data]

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert pytest.approx(run1[0].item()) == pytest.approx(run2[0].item())
    # Second step should not
    assert run1[1] != run2[1]


def test_weight_drop():
    input_ = torch.randn(2, 1, 10)

    wd_lstm = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)
    run1 = [x.sum() for x in wd_lstm(input_)[0].data]
    run2 = [x.sum() for x in wd_lstm(input_)[0].data]

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert pytest.approx(run1[0].item()) == pytest.approx(run2[0].item())
    # Second step should not
    assert run1[1] != run2[1]


def test_weight_drop_zero():
    input_ = torch.randn(2, 1, 10)

    wd_lstm = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.0)
    run1 = [x.sum() for x in wd_lstm(input_)[0].data]
    run2 = [x.sum() for x in wd_lstm(input_)[0].data]

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert pytest.approx(run1[0].item()) == pytest.approx(run2[0].item())
    # Second step should not
    assert pytest.approx(run1[1].item()) == pytest.approx(run2[1].item())
