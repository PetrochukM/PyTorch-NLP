# TODO: Use SRU / CNN Encoder / Embeddings etc...
# in a sequence to label
# on the IMBD dataset

from functools import partial

import argparse
import logging
import os

from torch.nn.modules.loss import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import torch
import pandas as pd

from lib.checkpoint import Checkpoint
from lib.configurable import add_config
from lib.configurable import configurable
from lib.datasets import count
from lib.datasets import simple_qa_predicate
from lib.metrics import get_accuracy
from lib.metrics import print_bucket_accuracy
from lib.metrics import print_random_sample
from lib.nn import SeqToLabel
from lib.optim import Optimizer
from lib.samplers import BucketBatchSampler
from lib.text_encoders import IdentityEncoder
from lib.text_encoders import MosesEncoder
from lib.utils import collate_fn
from lib.utils import get_log_directory_path
from lib.utils import get_total_parameters
from lib.utils import init_logging
from lib.utils import setup_training

Adam.__init__ = configurable(Adam.__init__)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 80)

DEFAULT_HYPERPARAMETERS = {
    'lib': {
        'nn': {
            'seq_encoder.SeqEncoder.__init__': {
                'bidirectional': True,
                'embedding_size': 16,
                'n_layers': 1,
                'rnn_cell': 'lstm',
                'embedding_dropout': 0.0,
                'rnn_variational_dropout': 0.0,
                'rnn_dropout': 0.4,
            },
            'seq_to_label.SeqToLabel': {
                'decode_dropout': 0.2,
                'rnn_size': 16,
                'rnn_cell': 'lstm',
            }
        },
        'optim.Optimizer.__init__': {
            'max_grad_norm': 1.0,
        }
    },
    'torch.optim.Adam.__init__': {
        'lr': 0.001,
        'weight_decay': 0,
    },
    'examples.train_seq_to_label.train': {
        'dataset': count,
        'random_seed': 123,
        'epochs': 2,
        'train_max_batch_size': 16,
        'dev_max_batch_size': 128
    }
}

add_config(DEFAULT_HYPERPARAMETERS)


@configurable
def train(
        log_directory,  # Logs experiments, checkpoints, etc are saved
        dataset,
        random_seed,
        epochs,
        train_max_batch_size,
        dev_max_batch_size,
        checkpoint_path=None,
        device=None):
    checkpoint = setup_training(dataset, checkpoint_path, log_directory, device, random_seed)

    # Init Dataset
    train_dataset, dev_dataset = dataset(train=True, dev=True, test=False)
    logger.info('Num Training Data: %d', len(train_dataset))
    logger.info('Num Development Data: %d', len(dev_dataset))

    # Init Encoders and encode dataset
    if checkpoint:
        text_encoder, label_encoder = checkpoint.input_encoder, checkpoint.output_encoder
    else:
        text_encoder = MosesEncoder(train_dataset['text'])
        label_encoder = IdentityEncoder(train_dataset['label'])
    for dataset in [train_dataset, dev_dataset]:
        for row in dataset:
            row['text'] = text_encoder.encode(row['text'])
            row['label'] = label_encoder.encode(row['label'])

    # Init Model
    if checkpoint:
        model = checkpoint.model
    else:
        model = SeqToLabel(text_encoder.vocab_size, label_encoder.vocab_size)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

    logger.info('Model:\n%s', model)
    logger.info('Total Parameters: %d', get_total_parameters(model))

    if torch.cuda.is_available():
        model.cuda(device_id=device)

    # Init Loss
    criterion = NLLLoss(size_average=False)
    if torch.cuda.is_available():
        criterion.cuda()

    # Init Optimizer
    # https://github.com/pytorch/pytorch/issues/679
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))

    # collate function merges rows into tensor batches
    collate_fn_partial = partial(collate_fn, input_key='text', output_key='label', sort_key='text')

    def prepare_batch(batch, train=False):
        # Prepare batch for model
        text, text_lengths = batch['text']
        label, _ = batch['label']
        if torch.cuda.is_available():
            text, text_lengths = text.cuda(async=True), text_lengths.cuda(async=True)
            label = label.cuda(async=True)
        return Variable(text, volatile=not train), text_lengths, Variable(label, volatile=not train)

    # Train!
    for epoch in range(epochs):
        logger.info('Epoch %d', epoch)
        model.train(mode=True)
        batch_sampler = BucketBatchSampler(train_dataset, lambda r: r['text'].size()[0],
                                           train_max_batch_size)
        train_iterator = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_partial,
            pin_memory=torch.cuda.is_available(),
            num_workers=0)
        for batch in tqdm(train_iterator):
            text, text_lengths, label = prepare_batch(batch, True)
            optimizer.zero_grad()
            output = model(text, text_lengths)
            loss = criterion(output, label) / label.size()[0]

            # Backward propagation
            loss.backward()
            optimizer.step()

        Checkpoint.save(
            log_directory=log_directory,
            model=model,
            optimizer=optimizer,
            input_text_encoder=text_encoder,
            output_text_encoder=label_encoder,
            device=device)

        # Evaluate
        model.train(mode=False)
        texts, labels, outputs = [], [], []
        dev_iterator = DataLoader(
            dev_dataset,
            batch_size=dev_max_batch_size,
            collate_fn=collate_fn_partial,
            pin_memory=torch.cuda.is_available(),
            num_workers=0)
        total_loss = 0
        for batch in dev_iterator:
            text, text_lengths, label = prepare_batch(batch)
            output = model(text, text_lengths)
            total_loss += criterion(output, label).data[0]
            # Prevent memory leak by moving output from variable to tensor
            texts.extend(text.data.cpu().transpose(0, 1).split(split_size=1, dim=0))
            labels.extend(label.data.cpu().split(split_size=1, dim=0))
            outputs.extend(output.data.cpu().split(split_size=1, dim=0))

        optimizer.update(total_loss / len(dev_dataset), epoch)
        buckets = [label_encoder.decode(label) for label in labels]
        print_random_sample(texts, labels, outputs, text_encoder, label_encoder)
        print_bucket_accuracy(buckets, labels, outputs, print_=True)
        logger.info('Loss: %.03f', total_loss / len(dev_dataset))
        get_accuracy(labels, outputs, print_=True)

    # TODO: Return the best loss if hyperparameter tunning.

    # TODO: Figure out a good abstraction for evaluation.
    # TODO: In class, I'll add a classification model and try to get it running.


if __name__ == '__main__':
    """
    Simple Questions Predicate CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_path', type=str, default=None, help="Path to checkpoint to resume training.")
    args = parser.parse_args()

    if args.checkpoint_path:
        log_directory = os.path.dirname(args.checkpoint_path)
    else:
        log_directory = get_log_directory_path('seq_to_seq')
    log_directory = init_logging(log_directory)
    logger = logging.getLogger(__name__)
    train(checkpoint_path=args.checkpoint_path, log_directory=log_directory)
