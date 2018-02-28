#!/usr/bin/env python 3.6
import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get(
                "encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('torchnlp', '__init__.py')
LONG_DESCRIPTION = read('README.md')

setup_info = dict(
    # Metadata
    name='pytorch-nlp',
    version=VERSION,
    author='Michael Petrochuk',
    author_email='petrochukm@gmail.com',
    url='https://github.com/Deepblue129/PytorchNLP',
    description='Text utilities and datasets for PyTorch',
    long_description=LONG_DESCRIPTION,
    license='BSD',
    install_requires=[
        'dill', 'scikit-optimize', 'nltk', 'numpy', 'pandas', 'tqdm', 'wrapt', 'ujson', 'spacy'
    ],

    # Package info
    packages=find_packages(exclude=('tests',)),
    zip_safe=True,
)

setup(**setup_info)
