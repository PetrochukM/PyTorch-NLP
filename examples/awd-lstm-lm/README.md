`awd-lstm-lm` set the state-of-the-art in word level perplexities in 2017. With PyTorch NLP, we show that in 30 minutes, we were able to reduce the footprint of this repository by 4 files (185 lines of code). We employ the use of the [datasets package](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html), [IdentityEncoder module](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.encoders.text.html#torchnlp.encoders.text.IdentityEncoder), [BPTTBatchSampler module](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.samplers.html#torchnlp.samplers.BPTTBatchSampler), [LockedDropout module](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.nn.html#torchnlp.nn.LockedDropout) and [WeightDrop module](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.nn.html#torchnlp.nn.WeightDrop)


Below is the original README from the repository:

-------------------------------------------------------------------------------

# LSTM and QRNN Language Model Toolkit

This repository contains the code used for two [Salesforce Research](https://einstein.ai/) papers:
+ [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
+ [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240)
This code was originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model).

The model comes with instructions to train:
+ word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets

+ character level language models over the Penn Treebank (PTBC) and Hutter Prize dataset (enwik8)

The model can be composed of an LSTM or a [Quasi-Recurrent Neural Network](https://github.com/salesforce/pytorch-qrnn/) (QRNN) which is two or more times faster than the cuDNN LSTM in this setup while achieving equivalent or better accuracy.

+ Install PyTorch 0.3
+ Run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets
+ Train the base model using `main.py`

If you use this code or our results in your research, please cite as appropriate:

```
@article{merityRegOpt,
  title={{Regularizing and Optimizing LSTM Language Models}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1708.02182},
  year={2017}
}
```

```
@article{merityAnalysis,
  title={{An Analysis of Neural Language Modeling at Multiple Scales}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1803.08240},
  year={2018}
}
```

## Software Requirements

Python 3 and PyTorch 0.3 are required for the current codebase.

Included below are hyper parameters to get equivalent or better results to those included in the original paper.

If you need to use an earlier version of the codebase, the original code and hyper parameters accessible at the [PyTorch==0.1.12](https://github.com/salesforce/awd-lstm-lm/tree/PyTorch%3D%3D0.1.12) release, with Python 3 and PyTorch 0.1.12 are required.
If you are using Anaconda, installation of PyTorch 0.1.12 can be achieved via:
`conda install pytorch=0.1.12 -c soumith`.

## Experiments

The codebase was modified during the writing of the paper, preventing exact reproduction due to minor differences in random seeds or similar.
We have also seen exact reproduction numbers change when changing underlying GPU.
The guide below produces results largely similar to the numbers reported.

For data setup, run `./getdata.sh`.
This script collects the Mikolov pre-processed Penn Treebank and the WikiText-2 datasets and places them in the `data` directory.

Next, decide whether to use the QRNN or the LSTM as the underlying recurrent neural network model.
The QRNN is many times faster than even Nvidia's cuDNN optimized LSTM (and dozens of times faster than a naive LSTM implementation) yet achieves similar or better results than the LSTM for many word level datasets.
At the time of writing, the QRNN models use the same number of parameters and are slightly deeper networks but are two to four times faster per epoch and require less epochs to converge.

The QRNN model uses a QRNN with convolutional size 2 for the first layer, allowing the model to view discrete natural language inputs (i.e. "New York"), while all other layers use a convolutional size of 1.

**Finetuning Note:** Fine-tuning modifies the original saved model `model.pt` file - if you wish to keep the original weights you must copy the file.

**Pointer note:** BPTT just changes the length of the sequence pushed onto the GPU but won't impact the final result.

### Character level enwik8 (PTB) with LSTM

+ `python -u main.py --epochs 50 --nlayers 3 --emsize 400 --nhid 1840 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.4 --wdrop 0.2 --wdecay 1.2e-6 --bptt 200 --batch_size 128 --optimizer adam --lr 1e-3 --data data/enwik8 --save ENWIK8.pt --when 25 35`

### Character level Penn Treebank (PTB) with LSTM

+ `python -u main.py --epochs 500 --nlayers 3 --emsize 200 --nhid 1000 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.25 --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 1.2e-6 --bptt 150 --batch_size 128 --optimizer adam --lr 2e-3 --data data/pennchar --save PTBC.pt --when 300 400`

### Word level WikiText-103 (PTB) with QRNN

+ `python -u main.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 0 --bptt 140 --batch_size 60 --optimizer adam --lr 1e-3 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12`

### Word level Penn Treebank (PTB) with LSTM

The instruction below trains a PTB model that without finetuning achieves perplexities of approximately `61.2` / `58.8` (validation / testing).

+ `python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt`

### Word level Penn Treebank (PTB) with QRNN

The instruction below trains a QRNN model that without finetuning achieves perplexities of approximately `60.6` / `58.3` (validation / testing).

+ `python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB.pt`

### Word level WikiText-2 (WT2) with LSTM
The instruction below trains a PTB model that without finetuning achieves perplexities of approximately `68.7` / `65.6` (validation / testing).

+ `python main.py --epochs 750 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --seed 1882`

### Word level WikiText-2 (WT2) with QRNN

The instruction below will a QRNN model that without finetuning achieves perplexities of approximately `69.3` / `66.8` (validation / testing).
Better numbers are likely achievable but the hyper parameters have not been extensively searched. These hyper parameters should serve as a good starting point however.

+ `python -u main.py --epochs 500 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550  --nlayers 4 --seed 4002 --model QRNN --wdrop 0.1 --batch_size 40 --save WT2.pt`

## Speed

For speed regarding character-level PTB and enwik8 or word-level WikiText-103, refer to the relevant paper.

The default speeds for the models during training on an NVIDIA Quadro GP100:

+ Penn Treebank (batch size 20): LSTM takes 65 seconds per epoch, QRNN takes 28 seconds per epoch
+ WikiText-2 (batch size 20): LSTM takes 180 seconds per epoch, QRNN takes 90 seconds per epoch

The default QRNN models can be far faster than the cuDNN LSTM model, with the speed-ups depending on how much of a bottleneck the RNN is. The majority of the model time above is now spent in softmax or optimization overhead (see [PyTorch QRNN discussion on speed](https://github.com/salesforce/pytorch-qrnn#speed)).

Speeds are approximately three times slower on a K80. On a K80 or other memory cards with less memory you may wish to enable [the cap on the maximum sampled sequence length](https://github.com/salesforce/awd-lstm-lm/blob/ef9369d277f8326b16a9f822adae8480b6d492d0/main.py#L131) to prevent out-of-memory (OOM) errors, especially for WikiText-2.

If speed is a major issue, SGD converges more quickly than our non-monotonically triggered variant of ASGD though achieves a worse overall perplexity.

### Details of the QRNN optimization

For full details, refer to the [PyTorch QRNN repository](https://github.com/salesforce/pytorch-qrnn).

### Details of the LSTM optimization

All the augmentations to the LSTM, including our variant of [DropConnect (Wan et al. 2013)](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) termed weight dropping which adds recurrent dropout, allow for the use of NVIDIA's cuDNN LSTM implementation.
PyTorch will automatically use the cuDNN backend if run on CUDA with cuDNN installed.
This ensures the model is fast to train even when convergence may take many hundreds of epochs.
