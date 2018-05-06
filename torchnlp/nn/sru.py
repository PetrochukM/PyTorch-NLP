from collections import namedtuple

import logging
import math
import os
import platform
import re
import subprocess

from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.utils as utils

logger = logging.getLogger(__name__)


# This SRU version implements its own cuda-level optimization,
# so it requires that:
# 1. `cupy` and `pynvrtc` python package installed.
# 2. pytorch is built with cuda support.
# 3. library path set: export LD_LIBRARY_PATH=<cuda lib path>.
def check_sru_requirement(abort=False):  # pragma: no cover
    """
    Return True if check pass; if check fails and abort is True,
    raise an Exception, othereise return False.
    """
    # Check 1.
    try:
        if platform.system() == 'Windows':
            subprocess.check_output('pip freeze | findstr cupy', shell=True)
            subprocess.check_output('pip freeze | findstr pynvrtc', shell=True)
        else:  # Unix-like systems
            subprocess.check_output('pip freeze | grep -w cupy', shell=True)
            subprocess.check_output('pip freeze | grep -w pynvrtc', shell=True)
    except subprocess.CalledProcessError:
        if not abort:
            return False
        raise AssertionError("Using SRU requires 'cupy' and 'pynvrtc' "
                             "python packages installed.")

    # Check 2.
    if torch.cuda.is_available() is False:
        if not abort:
            return False
        raise AssertionError("Using SRU requires pytorch built with cuda.")

    # Check 3.
    pattern = re.compile(".*cuda/lib.*")
    ld_path = os.getenv('LD_LIBRARY_PATH', "")
    if re.match(pattern, ld_path) is None:
        if not abort:
            return False
        raise AssertionError("Using SRU requires setting cuda lib path, e.g. "
                             "export LD_LIBRARY_PATH=/usr/local/cuda/lib64.")

    return True


if check_sru_requirement():  # pragma: no cover
    from cupy.cuda import function
    from pynvrtc.compiler import Program

    # This cuda() is important, it sets up device to use.
    tmp_ = torch.rand(1, 1).cuda()

    SRU_CODE = open('sru.cu').read()
    SRU_PROG = Program(SRU_CODE.encode('utf-8'), 'sru_prog.cu'.encode('utf-8'))
    SRU_PTX = SRU_PROG.compile()

    Stream = namedtuple('Stream', ['ptr'])


class _SRUComputeGPU(Function):  # pragma: no cover
    """
    Compute SRU with a GPU employing the use of CUDA.
    """

    _DEVICE2FUNC = {}

    def __init__(self, activation_type, d_out, bidirectional=False, scale_x=1):
        super(_SRUComputeGPU, self).__init__()
        # Raise error if requirements are not installed
        check_sru_requirement(abort=True)
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional
        self.scale_x = scale_x

    def compile_functions(self):
        device = torch.cuda.current_device()
        logger.info('SRU loaded for gpu {}'.format(device))
        mod = function.Module()
        mod.load(bytes(SRU_PTX.encode()))
        fwd_func = mod.get_function('sru_fwd')
        bwd_func = mod.get_function('sru_bwd')
        bifwd_func = mod.get_function('sru_bi_fwd')
        bibwd_func = mod.get_function('sru_bi_bwd')

        current_stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)

        self._DEVICE2FUNC[device] = (current_stream, fwd_func, bifwd_func, bwd_func, bibwd_func)
        return current_stream, fwd_func, bifwd_func, bwd_func, bibwd_func

    def get_functions(self):
        res = self._DEVICE2FUNC.get(torch.cuda.current_device(), None)
        return res if res else self.compile_functions()

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d * bidir) if x.dim() == 3 else (batch, d * bidir)
        c = x.new(*size)
        h = x.new(*size)

        scale_x = self.scale_x
        if k_ == 3:
            x_ptr = x.contiguous() * scale_x if scale_x != 1 else x.contiguous()
            x_ptr = x_ptr.data_ptr()
        else:
            x_ptr = 0

        stream, fwd_func, bifwd_func, _, _ = self.get_functions()
        FUNC = fwd_func if not self.bidirectional else bifwd_func
        FUNC(
            args=[
                u.contiguous().data_ptr(), x_ptr,
                bias.data_ptr(),
                init_.contiguous().data_ptr(),
                mask_h.data_ptr() if mask_h is not None else 0, length, batch, d, k_,
                h.data_ptr(),
                c.data_ptr(), self.activation_type
            ],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=stream)

        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.cat((c[-1, :, :d], c[0, :, d:]), dim=1)
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        scale_x = self.scale_x
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1

        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * bidir)
        grad_init = x.new(batch, d * bidir)

        # Normal use
        grad_x = x.new(*x.size()) if k_ == 3 else None

        if k_ == 3:
            x_ptr = x.contiguous() * scale_x if scale_x != 1 else x.contiguous()
            x_ptr = x_ptr.data_ptr()
        else:
            x_ptr = 0

        stream, _, _, bwd_func, bibwd_func = self.get_functions()
        FUNC = bwd_func if not self.bidirectional else bibwd_func
        FUNC(
            args=[
                u.contiguous().data_ptr(), x_ptr,
                bias.data_ptr(),
                init_.contiguous().data_ptr(),
                mask_h.data_ptr() if mask_h is not None else 0,
                c.data_ptr(),
                grad_h.contiguous().data_ptr(),
                grad_last.contiguous().data_ptr(), length, batch, d, k_,
                grad_u.data_ptr(),
                grad_x.data_ptr() if k_ == 3 else 0,
                grad_bias.data_ptr(),
                grad_init.data_ptr(), self.activation_type
            ],
            block=(thread_per_block, 1, 1),
            grid=(num_block, 1, 1),
            stream=stream)

        if k_ == 3 and scale_x != 1:
            grad_x.mul_(scale_x)
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


def _sru_compute_cpu(activation_type, d, bidirectional=False, scale_x=1):
    """CPU version of the core SRU computation.

    Has the same interface as _SRUComputeGPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    def sru_compute_cpu(u, x, bias, init=None, mask_h=None):
        bidir = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // d // bidir

        if mask_h is None:
            mask_h = 1

        u = u.view(length, batch, bidir, d, k)

        x_tilde = u[..., 0]

        forget_bias, reset_bias = bias.view(2, bidir, d)
        forget = (u[..., 1] + forget_bias).sigmoid()
        reset = (u[..., 2] + reset_bias).sigmoid()

        if k == 3:
            x_prime = x.view(length, batch, bidir, d)
            x_prime = x_prime * scale_x if scale_x != 1 else x_prime
        else:
            x_prime = u[..., 3]

        h = x.new_empty(length, batch, bidir, d)

        if init is None:
            c_init = x.new_zeros(batch, bidir, d)
        else:
            c_init = init.view(batch, bidir, d)

        c_final = []
        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)

            c_prev = c_init[:, di, :]
            for t in time_seq:
                c_t = (c_prev - x_tilde[t, :, di, :]) * forget[t, :, di, :] + x_tilde[t, :, di, :]
                c_prev = c_t

                if activation_type == 0:
                    g_c_t = c_t
                elif activation_type == 1:
                    g_c_t = c_t.tanh()
                elif activation_type == 2:
                    g_c_t = nn.functional.relu(c_t)
                elif activation_type == 3:
                    g_c_t = nn.functional.selu(c_t)

                h[t, :, di, :] = ((g_c_t * mask_h - x_prime[t, :, di, :]) * reset[t, :, di, :] +
                                  x_prime[t, :, di, :])

            c_final.append(c_t)

        return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(batch, -1)

    return sru_compute_cpu


class SRUCell(nn.Module):
    """ Simple Recurrent Unit (SRU) cell

    A recurrent cell that simplifies the computation and exposes more parallelism. In SRU, the
    majority of computation for each step is independent of the recurrence and can be easily
    parallelized. SRU is as fast as a convolutional layer and 5-10x faster than an optimized
    LSTM implementation.

    **Thank you** to taolei87 for their initial implementation of :class:`SRU`. Here is
    their `License
    <https://github.com/taolei87/sru/blob/master/LICENSE>`__.

    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        nonlinearity: The non-linearity to use ['tanh'|'relu'|'selu'|''].
        highway_bias: If ``False``, then the layer does not use highway bias weights b_ih and b_hh.
        stacked_dropout: If non-zero, introduces a stacked dropout layer on the outputs of each
            RNN layer except the last layer
        recurrent_dropout: If non-zero, introduces a recurrent dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If ``True``, becomes a bidirectional RNN.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 nonlinearity='tanh',
                 highway_bias=0,
                 stacked_dropout=0,
                 recurrent_dropout=0,
                 bidirectional=False):
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.stacked_dropout = stacked_dropout
        self.bidirectional = bidirectional
        self.highway_bias = highway_bias
        self.activation_type = 0
        self.nonlinearity = nonlinearity
        if nonlinearity.lower() == '':
            self.activation_type = 0
        elif nonlinearity.lower() == 'tanh':
            self.activation_type = 1
        elif nonlinearity.lower() == 'relu':
            self.activation_type = 2
        elif nonlinearity.lower() == 'selu':
            self.activation_type = 3
        else:
            raise ValueError(
                'WARNING: Activation functions must be either: `relu`, `tanh` or `selu`')

        out_size = hidden_size * 2 if bidirectional else hidden_size
        k = 4 if input_size != out_size else 3
        self.k = k
        self.size_per_dir = hidden_size * k
        self.weight = nn.Parameter(
            torch.Tensor(input_size, self.size_per_dir * 2 if bidirectional else self.size_per_dir))
        self.bias = nn.Parameter(
            torch.Tensor(hidden_size * 4 if bidirectional else hidden_size * 2))
        self.init_weight()

    def init_weight(self, rescale=True):
        # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
        val_range = (3.0 / self.input_size)**0.5
        self.weight.data.uniform_(-val_range, val_range)

        # initialize bias
        self.bias.data.zero_()
        bias_val, hidden_size = self.highway_bias, self.hidden_size
        if self.bidirectional:
            self.bias.data[hidden_size * 2:].zero_().add_(bias_val)
        else:
            self.bias.data[hidden_size:].zero_().add_(bias_val)

        self.scale_x = 1

        if rescale:
            self.scale_x = (1 + math.exp(bias_val) * 2)**0.5

            # re-scale weights in case there's dropout
            w = self.weight.data.view(self.input_size, -1, self.hidden_size, self.k)
            if self.stacked_dropout > 0:
                w[:, :, :, 0].mul_((1 - self.stacked_dropout)**0.5)
            if self.recurrent_dropout > 0:
                w.mul_((1 - self.recurrent_dropout)**0.5)
            if self.k == 4:
                w[:, :, :, 3].mul_(self.scale_x)

    def forward(self, input_, c0=None):
        """
        Args:
            input_ (seq_length, batch, input_size): Tensor containing input features.
            c0 (batch, hidden_size * num_directions): Tensor containing the initial
                hidden state for each element in the batch.
        Returns:
            output (seq_length, batch, hidden_size * num_directions): Tensor containing output
                features from the last layer of the RNN
            c0 (batch, hidden_size * num_directions): Tensor containing the initial
                hidden state for each element in the batch.
        """
        assert input_.dim() == 2 or input_.dim() == 3
        input_size, hidden_size = self.input_size, self.hidden_size
        batch = input_.size(-2)
        if c0 is None:
            c0 = input_.new_zeros(batch, hidden_size if not self.bidirectional else hidden_size * 2)

        if self.training and (self.recurrent_dropout > 0):
            mask = self.get_dropout_mask_((batch, input_size), self.recurrent_dropout)
            x = input_ * mask.expand_as(input_)
        else:
            x = input_

        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, input_size)
        u = x_2d.mm(self.weight)

        if input_.is_cuda:  # pragma: no cover
            SRU_Compute = _SRUComputeGPU(self.activation_type, hidden_size, self.bidirectional,
                                         self.scale_x)
        else:
            SRU_Compute = _sru_compute_cpu(self.activation_type, hidden_size, self.bidirectional,
                                           self.scale_x)

        mask_h = None
        if self.training and (self.stacked_dropout > 0):
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, hidden_size * bidir), self.stacked_dropout)

        return SRU_Compute(u, input_, self.bias, c0, mask_h)

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return w.new(*size).bernoulli_(1 - p).div_(1 - p)

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        if self.bias is not True:
            s += ', highway_bias={highway_bias}'
        if self.stacked_dropout != 0:
            s += ', stacked_dropout={stacked_dropout}'
        if self.recurrent_dropout != 0:
            s += ', recurrent_dropout={recurrent_dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class SRU(nn.Module):
    """ Simple Recurrent Unit (SRU) module

    A recurrent unit that simplifies the computation and exposes more parallelism. In SRU, the
    majority of computation for each step is independent of the recurrence and can be easily
    parallelized. SRU is as fast as a convolutional layer and 5-10x faster than an optimized
    LSTM implementation.

    **Thank you** to taolei87 for their initial implementation of :class:`SRU`. Here is
    their `SRU_License`_.

    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu'|'selu'].
        highway_bias: If ``False``, then the layer does not use highway bias weights b_ih and b_hh.
        stacked_dropout: If non-zero, introduces a stacked dropout layer on the outputs of each
            RNN layer except the last layer
        recurrent_dropout: If non-zero, introduces a recurrent dropout layer on the outputs of each
            RNN layer except the last layer
        bidirectional: If ``True``, becomes a bidirectional RNN.

    .. _SRU_License:
      https://github.com/taolei87/sru/blob/de3a97f4173f2fa00a949ae3aab31cb9b2f49b65/LICENSE
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=2,
            nonlinearity='tanh',
            highway_bias=0,
            stacked_dropout=0,
            recurrent_dropout=0,
            bidirectional=False,
    ):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stacked_dropout = stacked_dropout
        self.recurrent_dropout = recurrent_dropout
        self.rnn_list = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size
        self.nonlinearity = nonlinearity
        self.highway_bias = highway_bias

        for i in range(num_layers):
            layer = SRUCell(
                input_size=self.input_size if i == 0 else self.out_size,
                hidden_size=self.hidden_size,
                stacked_dropout=stacked_dropout if i + 1 != num_layers else 0,
                recurrent_dropout=recurrent_dropout,
                bidirectional=bidirectional,
                nonlinearity=nonlinearity,
                highway_bias=highway_bias)
            self.rnn_list.append(layer)

    def forward(self, input_, c0=None):
        """
        Args:
            input_ (seq_length, batch, input_size): Tensor containing input features.
            c0 (torch.num_layers, batch, hidden_size * num_directions): Tensor containing the
                initial hidden state for each element in the batch.
        Returns:
            output (seq_length, batch, hidden_size * num_directions): Tensor containing output
                features from the last layer of the RNN
            c0 (num_layers, batch, hidden_size * num_directions): Tensor containing the initial
                hidden state for each element in the batch.
        """
        is_packed = isinstance(input_, utils.rnn.PackedSequence)
        if is_packed:
            input_, lengths = utils.rnn.pad_packed_sequence(input_, batch_first=False)

        assert input_.dim() == 3  # (len, batch, input_size)
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = input_.new_zeros(input_.size(1), self.hidden_size * dir_)
            c0 = [zeros for i in range(self.num_layers)]
        else:
            assert c0.dim() == 3  # (num_layers, batch, hidden_size*dir_)
            c0 = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        prevx = input_
        lstc = []
        for i, rnn in enumerate(self.rnn_list):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)

        if is_packed:
            prevx = utils.rnn.pack_padded_sequence(prevx, lengths, batch_first=False)

        return prevx, torch.stack(lstc)

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        s += ', nonlinearity={nonlinearity}'
        s += ', num_layers={num_layers}'
        if self.highway_bias is not True:
            s += ', highway_bias={highway_bias}'
        if self.stacked_dropout != 0:
            s += ', stacked_dropout={stacked_dropout}'
        if self.recurrent_dropout != 0:
            s += ', recurrent_dropout={recurrent_dropout}'
        if self.bidirectional:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
