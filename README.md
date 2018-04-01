<p align="center"><img width="55%" src="docs/_static/img/logo_horizontal_color.svg" /></p>

-------------------------------------------------------------------------------

PyTorch-NLP is a library for Natural Language Processing (NLP) in PyTorch. It's built with the very
latest research in mind and was designed from day one to support rapid prototyping. PyTorch-NLP
comes with **neural network modules** and [FastText pre-trained word vectors](http://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.embeddings.html#torchnlp.embeddings.FastText).
It features 9 text encoders for preprocessing, integration with **9 popular datasets** and samplers
to be used with PyTorch DataLoaders. It's open-source software, released under the BSD3 license. 

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-nlp.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/PyTorch-NLP/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/PyTorch-NLP) 
[![Documentation Status](	https://img.shields.io/readthedocs/pytorchnlp/latest.svg?style=flat-square)](http://pytorchnlp.readthedocs.io/en/latest/?badge=latest&style=flat-square)
[![Build Status](https://img.shields.io/travis/PetrochukM/PyTorch-NLP/master.svg?style=flat-square)](https://travis-ci.org/PetrochukM/PyTorch-NLP)
[![Gitter chat](https://img.shields.io/gitter/room/PyTorch-NLP/Lobby.svg?style=flat-square)](https://gitter.im/PyTorch-NLP?style=flat-square)

## Installation

Make sure you have Python 3.5+ and PyTorch 0.2.0 or newer. You can then install `pytorch-nlp` using
pip:

    pip install pytorch-nlp

### Optional Tokenizer Requirements

If you want to use English tokenizer from `SpaCy <http://spacy.io/>`, you need to install SpaCy and download its English model:

    pip install spacy
    python -m spacy download en_core_web_sm

Alternatively, you might want to use Moses tokenizer from `NLTK <http://nltk.org/>`. You have to install NLTK and download the data needed:

    pip install nltk
    python -m nltk.downloader perluniprops nonbreaking_prefixes


### Optional CUDA Requirements

Along with standard PyTorch CUDA requirements, if you want to use Simple Recurrent Unit (SRU) with
CUDA, you need to install `cupy` and `pynvrtc`:

    pip install cupy
    pip install pynvrtc
    
## Documentation 📖 

The complete documentation for PyTorch-NLP is available via [our ReadTheDocs website](https://pytorchnlp.readthedocs.io).

## Quickstart

Add PyTorch NLP to your project by following one the common use cases:

- From the [neural network package](http://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.nn.html),
  use a Simple Recurrent Unit (SRU), like so:

    ```python
    from torchnlp.nn import SRU
    import torch

    input_ = torch.autograd.Variable(torch.randn(6, 3, 10))
    sru = SRU(10, 20)

    # Apply a Simple Recurrent Unit to `input_`
    sru(input_) # RETURNS: (output [torch.FloatTensor of size 6x3x20], hidden_state [torch.FloatTensor of size 2x3x20])
    ```

- Load a [dataset](http://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html) like IMDB.

    ```python
    from torchnlp.datasets import imdb_dataset
    
    # Load the imdb training dataset
    train = imdb_dataset(train=True)
    train[0]  # RETURNS: {'text': 'For a movie that gets..', 'sentiment': 'pos'}
    ```
      
- Encode text into vectors with the [text encoders package](http://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.text_encoders.html).

    ```python
    from torchnlp.text_encoders import WhitespaceEncoder
    
    # Create a `WhitespaceEncoder` with a corpus of text
    encoder = WhitespaceEncoder(["now this ain't funny", "so don't you dare laugh"])
    
    # Encode and decode phrases
    encoder.encode("this ain't funny.") # RETURNS: torch.LongTensor([6, 7, 1])
    encoder.decode(encoder.encode("This ain't funny.")) # RETURNS: "this ain't funny."
    ```
    
- Load FastText, state-of-the-art English [embeddings](http://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.embeddings.html).

    ```python
    from torchnlp.embeddings import FastText
    
    vectors = FastText()
    # Load embeddings for any word as a `torch.FloatTensor`
    vectors['hello']  # RETURNS: [torch.FloatTensor of size 100]
    ```
    
- Compute the BLEU Score with the [metrics package](http://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.metrics.html).

    ```python
    from torchnlp.metrics import get_moses_multi_bleu
    
    hypotheses = ["The brown fox jumps over the dog 笑"]
    references = ["The quick brown fox jumps over the lazy dog 笑"]
    
    # Compute BLEU score with the official BLEU perl script
    get_moses_multi_bleu(hypotheses, references, lowercase=True)  # RETURNS: 47.9
    ```
    
PyTorch NLP is designed to be intuitive, linear in thought and easy to use. PyTorch NLP has minimal framework overhead.

## Contributing

We've released PyTorch-NLP because we found a lack of basic toolkits for NLP in PyTorch. We hope that other organizations can benefit from the project. We are thankful for any contributions from the community.

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/PyTorch-NLP/blob/master/Contributing.md) to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to PyTorch-NLP.

## Citing

If you find PyTorch NLP useful for an academic publication, then please use the following BibTeX to cite it:

```
@misc{pytorch-nlp,
  author = {Petrochuk, Michael},
  title = {PyTorch-NLP: Rapid PyTorch NLP prototyping tools for research},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PetrochukM/PyTorch-NLP}},
}
```

## Logo Credits

Thanks to [Chloe Yeo](http://www.yeochloe.com/) for her logo design.
