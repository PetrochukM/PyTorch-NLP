import os

from functools import lru_cache
from functools import partial
from itertools import product

import string
import random

from torch.autograd import Variable
from torch.nn.modules.loss import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.datasets.dataset import Dataset
from lib.nn import SeqToSeq
from lib.optim import Optimizer
from lib.samplers import BucketBatchSampler
from lib.text_encoders import PADDING_INDEX
from lib.text_encoders import WordEncoder
from lib.utils import device_default
from lib.utils import get_root_path
from lib.utils import pad_batch

DATA_DIR = os.path.join(get_root_path(), 'data')


def random_token(size=6, chars=string.ascii_uppercase + string.digits):
    """
    Args:
        size (int): size of the random token to create
        chars (list): list of characters to create random token
    Returns:
        Random token of `size` with characters from `chars`.
    """
    return ''.join(random.choice(chars) for x in range(size))


def random_tokens(size=100, token_generator=random_token):
    """
    Args:
        size (int): number of random tokens to create
    Returns:
        list of size `size` filled with unique random tokens
    """
    return [token_generator() for _ in range(size)]


def random_sequence(size=random.randint(1, 10), tokens_generator=random_tokens):
    """
    Args:
        size (int) length of the random sequence space separated
    Returns:
        (str) Random string of sequences space separated
    """
    return ' '.join(tokens_generator(size=size))


def random_dataset(input_key='input',
                   output_key='output',
                   input_generator=random_sequence,
                   output_generator=random_sequence,
                   input_encoder=WordEncoder,
                   output_encoder=WordEncoder,
                   size=random.randint(1, 100)):
    """
    Returns:
        (lib.datasets.Dataset) dataset over random data
    """
    rows = []
    for _ in range(size):
        row = {}
        row[input_key] = input_generator()
        row[output_key] = output_generator()
        rows.append(row)
    dataset = Dataset(rows)
    input_encoder = input_encoder(dataset[input_key])
    output_encoder = output_encoder(dataset[output_key])
    for row in dataset:
        row[input_key] = input_encoder.encode(row[input_key])
        row[output_key] = output_encoder.encode(row[output_key])
    return dataset, input_encoder, output_encoder


def kwargs_product(dict_):
    """
    Args:
        dict_ (dict): dict with possible kwargs values
    Returns:
        (iterable) iterable over all combinations of the dict of kwargs
    Usage:
        >>> list(dict_product(dict(number=[1,2], character='ab')))
        [{'character': 'a', 'number': 1},
        {'character': 'a', 'number': 2},
        {'character': 'b', 'number': 1},
        {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dict_, x)) for x in product(*dict_.values()))


def tensor(*args, type_=torch.LongTensor, max_=100, variable=True):  # noqa: E999
    """
    Args:
        type_ constructor for a tensor
    Returns:
        type_ [*args] filled with random numbers from a uniform distribution [0, max]
    """
    ret = type_(*args).random_(to=max_ - 1)
    if variable:
        ret = Variable(ret)
    return ret


def get_test_data_path():
    return os.path.join(get_root_path(), 'tests/data/')


def random_model(input_vocab_size,
                 output_vocab_size,
                 embedding_size=random.randint(1, 10),
                 rnn_size=random.randint(1, 10),
                 n_layers=random.randint(1, 10)):
    """
    Instantiate a model with random parameters

    TODO: Consider picking any random model instead of seq_to_seq with random parameters

    Returns:
        (lib.nn.SeqToSeq) model instantiated with random paramaters
    """
    # Model modules
    seq_to_seq = SeqToSeq(
        input_vocab_size,
        output_vocab_size,
        embedding_size=embedding_size,
        rnn_size=rnn_size,
        n_layers=n_layers)

    if torch.cuda.is_available():
        seq_to_seq.cuda()

    for param in seq_to_seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    return seq_to_seq


def random_iterator(device,
                    dataset,
                    input_encoder,
                    output_encoder,
                    input_key='input',
                    output_key='output',
                    batch_size=random.randint(1, 32),
                    train=True):
    """
    Args:
        device (None or int) device arg, -1 for CPU, None for GPU, and 0+ for GPU device ID
        dataset (torchtext.data.Dataset)
        batch_size (int)
    Returns:
        (torchtext.data.Iterator) iterator over random data
    """
    batch_sampler = BucketBatchSampler(dataset, lambda r: r[input_key].size()[0], batch_size)

    def collate_fn(batch):
        """ list of tensors to a batch variable """
        # PyTorch RNN requires sorting decreasing size
        batch = sorted(batch, key=lambda row: len(row[input_key]), reverse=True)
        input_batch, input_lengths = pad_batch([row[input_key] for row in batch])
        output_batch, output_lengths = pad_batch([row[output_key] for row in batch])

        def batch_to_variable(batch):
            # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
            return Variable(torch.stack(batch).t_().contiguous(), volatile=not train)

        # Async minibatch allocation for speed
        # Reference: http://timdettmers.com/2015/03/09/deep-learning-hardware-guide/
        cuda = lambda t: t.cuda(async=True) if torch.cuda.is_available() else t

        return (cuda(batch_to_variable(input_batch)), cuda(torch.LongTensor(input_lengths)),
                cuda(batch_to_variable(output_batch)), cuda(torch.LongTensor(output_lengths)))

    for batch in DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            num_workers=0):
        # Prepare batch for model
        input_, input_lengths = batch[input_key]
        output, output_lengths = batch[output_key]
        if torch.cuda.is_available():
            input_, input_lengths = input_.cuda(async=True), input_lengths.cuda(async=True)
            output, output_lengths = output.cuda(async=True), output_lengths.cuda(async=True)
        yield (Variable(input_, volatile=not train), input_lengths, Variable(
            output, volatile=not train), output_lengths)


@lru_cache(maxsize=1)
def random_args(train=True, batch_size=None):
    """
    Motivation:
        Set arguments pseudo randomly so testing files do not have to figure out how to set them.
    Args:
        training (bool): Set the model in MockModel to check for if training mode is True or False
        during inference.
    Returns:
        (dict) arguments that a model would take set pseudo randomly
    """
    # Some constants
    batch_size = random.randint(1, 3) if batch_size is None else batch_size
    num_batches = random.randint(1, 5)
    input_seq_len = random.randint(1, 4)
    output_seq_len = random.randint(1, 4)
    n_layers = random.randint(1, 4)
    rnn_size = random.randint(1, 4) * 2  # NOTE: Ensure it's divisible by 2 for BRNN
    epoch = random.randint(1, 10)
    embedding_size = random.randint(1, 20)
    input_key = 'input'
    output_key = 'output'
    device = device_default()

    # Some modules
    # NOTE: MockSeq2Seq returns constant sized tensors; therefore, our data should be cleanly
    # divisible `batch_size * num_batches` by batch_size

    dataset, input_encoder, output_encoder = random_dataset(
        input_key,
        output_key,
        input_generator=partial(random_sequence, input_seq_len),
        output_generator=partial(random_sequence, output_seq_len),
        size=batch_size * num_batches)
    iterator = random_iterator(
        device,
        dataset,
        input_encoder,
        output_encoder,
        input_key=input_key,
        output_key=output_key,
        batch_size=batch_size,
        train=train)
    model = random_model(
        input_vocab_size=input_encoder.vocab_size,
        output_vocab_size=output_encoder.vocab_size,
        embedding_size=embedding_size,
        rnn_size=rnn_size,
        n_layers=n_layers)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))
    # NOTE: +2 to include <s> and </s> for output tensor.
    mock_model = MockSeqToSeq(output_encoder.vocab_size, batch_size, n_layers, rnn_size,
                              output_seq_len + 2, train)
    criterion = NLLLoss(ignore_index=PADDING_INDEX, size_average=False)
    return {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'n_layers': n_layers,
        'rnn_size': rnn_size,
        'input_seq_len': input_seq_len,
        'output_seq_len': output_seq_len,
        'num_batches': num_batches,
        'epoch': epoch,
        'step': epoch * random.randint(1, 10),  # NOTE: Steps must be divisible by epoch
        'embedding_size': embedding_size,
        'input_encoder': input_encoder,
        'output_encoder': output_encoder,
        'input_vocab_size': input_encoder.vocab_size,
        'output_vocab_size': output_encoder.vocab_size,
        # torchtext.data.Dataset random datset with `seq_len` input and output sequences
        'dataset': dataset,
        # torchtext.data.Iterator over `dataset`
        'iterator': iterator,
        'experiments_directory': os.path.join(get_root_path(), 'tests/experiments/'),
        # NOTE: Random model used for testing checkpoints. Do not try to run this model because
        # it may never generate an EOS token and stop.
        # (lib.nn.seq_to_seq)
        'model': model,
        # (lib.optim.Optimizer)
        'optimizer': optimizer,
        'criterion': criterion,  # TODO: Consider having a random loss function
        # NOTE: Mock model is used for testing evaluation. A random model cannot be
        # used to test it because it may never return an EOS token.
        'mock_model': mock_model,
        # Device arg, -1 for CPU, None for GPU, and 0+ for GPU device ID
        'device': device
    }


class MockSeqToSeq(nn.Module):
    """
    Mock model for `lib.nn.seq_to_seq`

    Motivation: For evaluation, it's important we mock the model return values. A random model
    may never return due to never predicting the `EOS` token.
    """

    def __init__(self, vocab_size, batch_size, n_layers, rnn_size, output_seq_len, mode):
        """
        Args:
            vocab_size (int)
            batch_size (int)
            n_layers (int)
            rnn_size (int)
            output_seq_len (int)
            mode (bool) Used to test that SeqToSeq is run with the right training mode. `False` for
                inference and `True` for training.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_layers = rnn_size
        self.output_seq_len = output_seq_len
        self.mode = mode

    def forward(self, batch):
        # Check to ensure correct mode
        assert self.training == self.mode
        # Same return value as SeqToSeq where the RNN Cell is a GRU
        decoder_hidden = tensor(
            self.n_layers,
            self.batch_size,
            self.rnn_size,
            max_=self.vocab_size,
            type_=torch.FloatTensor)
        decoder_outputs = tensor(
            self.output_seq_len - 1,
            self.batch_size,
            self.vocab_size,
            max_=self.vocab_size,
            type_=torch.FloatTensor)
        decoder_outputs = decoder_outputs.view(-1, self.vocab_size)
        decoder_outputs = F.log_softmax(decoder_outputs)
        decoder_outputs = decoder_outputs.view(self.output_seq_len - 1, self.batch_size,
                                               self.vocab_size)
        return decoder_outputs, decoder_hidden, None


def get_confidence(prediction, vocab_size):
    """
    Given a prediction index and vocab_size create a confidence tensor.

    Args:
        prediction (int): int that was predicted
        vocab_size (int): size of vocabulary to predict from
    Returns:
        (list of floats):
            sum(list) == 1
            the highest float is at the index `prediction`
    """
    assert prediction < vocab_size, "Prediction must smaller than the vocab."
    assert prediction >= 0, "Prediction cannot be negative."

    # Get prediction confidence
    prediction_confidence = random.randint(51, 100)
    total = 100 - prediction_confidence

    # Get the rest of the confidences
    confidences = []
    for i in range(vocab_size - 1):
        if i == vocab_size - 2:
            confidence = total
        else:
            confidence = random.randint(0, total)
        total = total - confidence
        confidences.append(confidence)

    # Add then together
    ret = []
    for i in range(vocab_size):
        if i == prediction:
            ret.append(round(prediction_confidence / 100.0, 2))
        else:
            ret.append(round(confidences.pop() / 100.0, 2))

    assert round(sum(ret), 2) == 1

    return ret


def get_random_batch(output_seq_len, input_seq_len, batch_size, output_field, input_field):
    """
    Motivation: Get a random batch to compute the loss on.

    Args:
        output_seq_len (int): length of the sequence
        batch_size (int): size of the batch
        output_field (torchtext.data.Field): field used to process the output
    Returns:
        outputs (torch.FloatTensor [output_seq_len, batch_size, dictionary_size]): random outputs of a
            batch.
        targets (torch.LongTensor [seq_output_seq_lenlen, batch_size]): random expected output of a batch.
    """
    outputs = torch.stack([
        F.softmax(torch.randn(output_seq_len, len(output_field.vocab))) for _ in range(batch_size)
    ])
    targets = torch.stack([
        torch.LongTensor(
            [random.randint(0, len(output_field.vocab) - 1) for _ in range(output_seq_len)])
        for _ in range(batch_size)
    ])
    inputs = torch.stack([
        torch.LongTensor(
            [random.randint(0, len(input_field.vocab) - 1) for _ in range(input_seq_len)])
        for _ in range(batch_size)
    ])
    outputs = outputs.transpose(0, 1).contiguous()
    targets = targets.transpose(0, 1).contiguous()
    inputs = inputs.transpose(0, 1).contiguous()
    assert outputs.size() == (output_seq_len, batch_size, len(output_field.vocab))
    assert targets.size() == (output_seq_len, batch_size)
    assert inputs.size() == (input_seq_len, batch_size)
    return inputs, targets, outputs


def get_batch(predictions, targets, sources=None, vocab_size=3, is_list_of_tensors=True):
    """
    Given predictions and targets, create a batch that matches those.

    Args:
        predictions (list of lists): Batch size number of lists of sequences.
        targets (list of lists): Batch size number of lists of sequences.
        sources (list of lists, optional): Batch size number of lists of sequences.
    Returns:
        outputs (torch.Tensor seq_len x batch_size x dictionary_size): outputs of a batch.
        batch (MockBatch)
    """
    if sources is None:
        sources = targets

    assert len(predictions) == len(targets), "Targets batchsize should be the same as predictions"
    assert len(targets) == len(sources), "Sources batchsize should be the same as predictions"
    outputs = []
    for sequence in predictions:
        output = [get_confidence(prediction, vocab_size) for prediction in sequence]
        outputs.append(output)

    batch_size = len(predictions)
    output_seq_len = len(predictions[0])
    input_seq_len = len(sources[0])
    for i in range(batch_size):
        assert len(targets[i]) == output_seq_len, "Sizes need to be consistent"
        assert len(predictions[i]) == output_seq_len, "Sizes need to be consistent"

    if is_list_of_tensors:
        sources = [torch.LongTensor(row) for row in sources]
        targets = [torch.LongTensor(row) for row in targets]
        outputs = [torch.FloatTensor(row) for row in outputs]
    else:
        sources = torch.LongTensor(sources).transpose(0, 1).contiguous()
        targets = torch.LongTensor(targets).transpose(0, 1).contiguous()
        outputs = torch.FloatTensor(outputs).transpose(0, 1).contiguous()

        assert sources.size() == (input_seq_len, batch_size)
        assert targets.size() == (output_seq_len, batch_size)
        assert outputs.size() == (output_seq_len, batch_size, vocab_size)

    return sources, targets, outputs
