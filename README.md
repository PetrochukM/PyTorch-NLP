# Pytorch-NLP

[![codecov](https://codecov.io/gh/MPetrochuk/PyTorch-NLP/branch/master/graph/badge.svg)](https://codecov.io/gh/MPetrochuk/PyTorch-NLP) 
[![Documentation Status](https://readthedocs.org/projects/pytorchnlp/badge/?version=latest)](http://pytorchnlp.readthedocs.io/en/latest/?badge=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![Build Status](https://travis-ci.org/MPetrochuk/PyTorch-NLP.svg?branch=master)](https://travis-ci.org/MPetrochuk/PyTorch-NLP)
[![License](https://img.shields.io/pypi/l/Django.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Test
python3.6 -m "pytest" --cov-report=html:coverage --cov-report=term-missing --cov=lib -m "not slow" tests/


## Installation

Make sure you have Python 3.5+ and PyTorch 0.2.0 or newer. You can then install pytorch-nlp using pip:

    pip install pytorch-nlp

### Optional requirements

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`_, you need to install SpaCy and download its English model:

    pip install spacy
    python -m spacy download en_core_web_sm

Alternatively, you might want to use Moses tokenizer from `NLTK <http://nltk.org/>`_. You have to install NLTK and download the data needed:

    pip install nltk
    python -m nltk.downloader perluniprops nonbreaking_prefixes
    