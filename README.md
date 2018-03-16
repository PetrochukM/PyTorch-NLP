# PyTorch-NLP

PyTorch-NLP is a library for Natural Language Processing (NLP) in Python. It's built with the very
latest research in mind, and was designed from day one to support rapid prototyping. PyTorch-NLP
comes with pre-trained embeddings, samplers, dataset loaders, metrics, neural network modules
and text encoders. It's open-source software, released under the BSD3 license. 

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-nlp.svg)
[![codecov](https://codecov.io/gh/PetrochukM/PyTorch-NLP/branch/master/graph/badge.svg)](https://codecov.io/gh/PetrochukM/PyTorch-NLP) 
[![Documentation Status](https://readthedocs.org/projects/pytorchnlp/badge/?version=latest)](http://pytorchnlp.readthedocs.io/en/latest/?badge=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![Build Status](https://travis-ci.org/PetrochukM/PyTorch-NLP.svg?branch=master)](https://travis-ci.org/PetrochukM/PyTorch-NLP)
[![License](https://img.shields.io/pypi/l/pytorch-nlp.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Documentation 📖 

The complete documentation for PyTorch-NLP is available via [our ReadTheDocs website](https://pytorchnlp.readthedocs.io).

## Installation

Make sure you have Python 3.5+ and PyTorch 0.2.0 or newer. You can then install `pytorch-nlp` using
pip:

    pip install pytorch-nlp

### Optional requirements

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`_, you need to install SpaCy and download its English model:

    pip install spacy
    python -m spacy download en_core_web_sm

Alternatively, you might want to use Moses tokenizer from `NLTK <http://nltk.org/>`_. You have to install NLTK and download the data needed:

    pip install nltk
    python -m nltk.downloader perluniprops nonbreaking_prefixes

## Contributing

We've released PyTorch-NLP because we found a lack of basic tool kits for NLP in PyTorch. We hope that other organizations can benefit from the project. We are thankful for any contributions from the community.

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/PyTorch-NLP/blob/master/Contributing.md) to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to PyTorch-NLP.

## License

Docusaurus is [BSD3 licensed](./LICENSE).
