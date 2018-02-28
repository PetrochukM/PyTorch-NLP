#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# Exit immediately if a command exits with a non-zero status.
set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

# Install requirements via pip
pip install -r requirements.txt
pip install -r requirements_dev.txt

# SpaCy English web model
python -m spacy download en_core_web_sm

# NLTK data needed for Moses tokenizer
python -m nltk.downloader perluniprops nonbreaking_prefixes

# Install PyTorch Dependancies
pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
pip install torchvision
