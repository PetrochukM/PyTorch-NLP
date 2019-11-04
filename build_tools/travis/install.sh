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

# TODO: Add a script similar to Travis to test locally with virtual environment
# TODO: Add a script similar to RTD to test locally with virtual environment

# Install requirements via pip
pip install -r requirements.txt --progress-bar off

# Optional Requirements
pip install spacy --progress-bar off
pip install nltk --progress-bar off
pip install sacremoses --progress-bar off
pip install pandas --progress-bar off
pip install requests --progress-bar off

# SpaCy English web model
python -m spacy download en

# NLTK data needed for Moses tokenizer
python -m nltk.downloader perluniprops nonbreaking_prefixes

# Install PyTorch Dependancies
pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  --progress-bar off

