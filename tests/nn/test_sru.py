from torch.autograd import Variable
from torch.nn import utils

import torch
import numpy as np

from torchnlp.nn import SRUCell
from torchnlp.nn import SRU


def _options(func):
    for dropout in [0.0, 0.5]:
        func(stacked_dropout=dropout, recurrent_dropout=dropout)

    for nonlinearity in ['', 'selu', 'relu', 'tanh']:
        func(nonlinearity=nonlinearity)

    for bias in [True, False]:
        func(highway_bias=bias)

    for bidirectional in [True, False]:
        func(bidirectional=bidirectional)


def test_SRUCell_smoke():
    # This is just a smoke test.

    def evaluate(**kwargs):
        input_ = Variable(torch.randn(6, 3, 10))
        c0 = None
        sru = SRUCell(10, 20, **kwargs)
        str(sru)
        for i in range(6):
            output, c0 = sru(input_[i], c0)

        c0.sum().backward()

    _options(evaluate)


def test_SRU_smoke():
    # This is just a smoke test.

    def evaluate(**kwargs):
        input_ = Variable(torch.randn(6, 6, 3, 10))
        c0 = None
        sru = SRU(10, 20, **kwargs)
        str(sru)
        for i in range(6):
            output, c0 = sru(input_[i], c0)

        c0.sum().backward()

    _options(evaluate)


def test_sru_packed():
    sru = SRU(4, 4, bidirectional=True)

    x = Variable(torch.randn(3, 2, 4))
    lengths = [3, 3]

    h1, c1 = sru(x)

    pack = utils.rnn.pack_padded_sequence(x, lengths)
    h2, c2 = sru(pack)
    h2, _ = utils.rnn.pad_packed_sequence(h2)

    x = torch.cat([x, Variable(x.data.new(1, 2, 4).zero_())])
    pack = utils.rnn.pack_padded_sequence(x, lengths)
    h3, c3 = sru(pack)
    h3, _ = utils.rnn.pad_packed_sequence(h3)

    h3.mean().backward()

    h_eq = (h1 == h2) == (h1 == h3)
    c_eq = (c1 == c2) == (c1 == c3)

    assert h_eq.sum().data[0] == np.prod(h_eq.size()) and c_eq.sum().data[0] == np.prod(c_eq.size())
