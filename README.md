<p align="center"><img width="55%" src="docs/_static/img/logo.svg" /></p>

<h3 align="center">Basic Utilities for PyTorch Natural Language Processing (NLP)</h3>

PyTorch-NLP, or `torchnlp` for short, is a library of basic utilities for PyTorch
NLP. `torchnlp` extends PyTorch to provide you with
basic text data processing functions.

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-nlp.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/PyTorch-NLP/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/PyTorch-NLP)
[![Downloads](http://pepy.tech/badge/pytorch-nlp)](http://pepy.tech/project/pytorch-nlp)
[![Documentation Status](https://img.shields.io/readthedocs/pytorchnlp/latest.svg?style=flat-square)](http://pytorchnlp.readthedocs.io/en/latest/?badge=latest&style=flat-square)
[![Build Status](https://img.shields.io/travis/PetrochukM/PyTorch-NLP/master.svg?style=flat-square)](https://travis-ci.org/PetrochukM/PyTorch-NLP)
[![Twitter: PetrochukM](https://img.shields.io/twitter/follow/MPetrochuk.svg?style=social)](https://twitter.com/MPetrochuk)

_Logo by [Chloe Yeo](http://www.yeochloe.com/), Corporate Sponsorship by [WellSaid Labs](https://wellsaidlabs.com/)_

## Installation 🐾

Make sure you have Python 3.5+ and PyTorch 1.0+. You can then install `pytorch-nlp` using
pip:

```python
pip install pytorch-nlp
```

Or to install the latest code via:

```python
pip install git+https://github.com/PetrochukM/PyTorch-NLP.git
```

## Docs

The complete documentation for PyTorch-NLP is available
via [our ReadTheDocs website](https://pytorchnlp.readthedocs.io).

## Get Started

Within an NLP data pipeline, you'll want to implement these basic steps:

### 1. Load your Data 🐿

Load the IMDB dataset, for example:

```python
from torchnlp.datasets import imdb_dataset

# Load the imdb training dataset
train = imdb_dataset(train=True)
train[0]  # RETURNS: {'text': 'For a movie that gets..', 'sentiment': 'pos'}
```

Load a custom dataset, for example:

```python
from pathlib import Path

from torchnlp.download import download_file_maybe_extract

directory_path = Path('data/')
train_file_path = Path('trees/train.txt')

download_file_maybe_extract(
    url='http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip',
    directory=directory_path,
    check_files=[train_file_path])

open(directory_path / train_file_path)
```

Don't worry we'll handle caching for you!

### 2. Text to Tensor

Tokenize and encode your text as a tensor.

For example, a `WhitespaceEncoder` breaks
text into tokens whenever it encounters a whitespace character.

```python
from torchnlp.encoders.text import WhitespaceEncoder

loaded_data = ["now this ain't funny", "so don't you dare laugh"]
encoder = WhitespaceEncoder(loaded_data)
encoded_data = [encoder.encode(example) for example in loaded_data]
```

### 3. Tensor to Batch

With your loaded and encoded data in hand, you'll want to batch your dataset.

```python
import torch
from torchnlp.samplers import BucketBatchSampler
from torchnlp.utils import collate_tensors
from torchnlp.encoders.text import stack_and_pad_tensors

encoded_data = [torch.randn(2), torch.randn(3), torch.randn(4), torch.randn(5)]

train_sampler = torch.utils.data.sampler.SequentialSampler(encoded_data)
train_batch_sampler = BucketBatchSampler(
    train_sampler, batch_size=2, drop_last=False, sort_key=lambda i: encoded_data[i].shape[0])

batches = [[encoded_data[i] for i in batch] for batch in train_batch_sampler]
batches = [collate_tensors(batch, stack_tensors=stack_and_pad_tensors) for batch in batches]
```

PyTorch-NLP builds on top of PyTorch's existing `torch.utils.data.sampler`, `torch.stack`
and `default_collate` to support sequential inputs of varying lengths!

### 4. Training and Inference

With your batch in hand, you can use PyTorch to develop and train your model using gradient descent.
For example, check out [this example code](examples/snli/train.py) for training on the Stanford
Natural Language Inference (SNLI) Corpus.

## Last But Not Least

PyTorch-NLP has a couple more NLP focused utility packages to support you! 🤗

### Deterministic Functions

Now you've setup your pipeline, you may want to ensure that some functions run deterministically.
Wrap any code that's random, with `fork_rng` and you'll be good to go, like so:

```python
import random
import numpy
import torch

from torchnlp.random import fork_rng

with fork_rng(seed=123):  # Ensure determinism
    print('Random:', random.randint(1, 2**31))
    print('Numpy:', numpy.random.randint(1, 2**31))
    print('Torch:', int(torch.randint(1, 2**31, (1,))))
```

This will always print:

```text
Random: 224899943
Numpy: 843828735
Torch: 843828736
```

### Pre-Trained Word Vectors

Now that you've computed your vocabulary, you may want to make use of
pre-trained word vectors to set your embeddings, like so:

```python
import torch
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.word_to_vector import GloVe

encoder = WhitespaceEncoder(["now this ain't funny", "so don't you dare laugh"])

pretrained_embedding = GloVe(name='6B', dim=100, is_include=lambda w: w in set(encoder.vocab))
embedding_weights = torch.Tensor(encoder.vocab_size, pretrained_embedding.dim)
for i, token in enumerate(encoder.vocab):
    embedding_weights[i] = pretrained_embedding[token]
```

### Neural Networks Layers

For example, from the neural network package, apply the state-of-the-art `LockedDropout`:

```python
import torch
from torchnlp.nn import LockedDropout

input_ = torch.randn(6, 3, 10)
dropout = LockedDropout(0.5)

# Apply a LockedDropout to `input_`
dropout(input_) # RETURNS: torch.FloatTensor (6x3x10)
```

### Metrics

Compute common NLP metrics such as the BLEU score.

```python
from torchnlp.metrics import get_moses_multi_bleu

hypotheses = ["The brown fox jumps over the dog 笑"]
references = ["The quick brown fox jumps over the lazy dog 笑"]

# Compute BLEU score with the official BLEU perl script
get_moses_multi_bleu(hypotheses, references, lowercase=True)  # RETURNS: 47.9
```

### Help :question:

Maybe looking at longer examples may help you at [`examples/`](examples/).

Need more help? We are happy to answer your questions via [Gitter Chat](https://gitter.im/PyTorch-NLP)

## Contributing

We've released PyTorch-NLP because we found a lack of basic toolkits for NLP in PyTorch. We hope
that other organizations can benefit from the project. We are thankful for any contributions from
the community.

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/PyTorch-NLP/blob/master/CONTRIBUTING.md)
to learn about our development process, how to propose bugfixes and improvements, and how to build
and test your changes to PyTorch-NLP.

## Related Work

### [torchtext](https://github.com/pytorch/text)

torchtext and PyTorch-NLP differ in the architecture and feature set; otherwise, they are similar.
torchtext and PyTorch-NLP provide pre-trained word vectors, datasets, iterators and text encoders.
PyTorch-NLP also provides neural network modules and metrics. From an architecture standpoint,
torchtext is object orientated with external coupling while PyTorch-NLP is object orientated with
low coupling.

### [AllenNLP](https://github.com/allenai/allennlp)

AllenNLP is designed to be a platform for research. PyTorch-NLP is designed to be a lightweight toolkit.

## Authors

- [Michael Petrochuk](https://github.com/PetrochukM/) — Developer
- [Chloe Yeo](http://www.yeochloe.com/) — Logo Design

## Citing

If you find PyTorch-NLP useful for an academic publication, then please use the following BibTeX to
cite it:

```
@misc{pytorch-nlp,
  author = {Petrochuk, Michael},
  title = {PyTorch-NLP: Rapid Prototyping with PyTorch Natural Language Processing (NLP) Tools},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PetrochukM/PyTorch-NLP}},
}
```
