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

setup_info = dict(
    # Metadata
    name='pytorch-nlp',
    version=VERSION,
    author='Michael Petrochuk',
    author_email='petrochukm@gmail.com',
    url='https://github.com/Deepblue129/PytorchNLP',
    description='Text utilities and datasets for PyTorch',
    long_description="PyTorch-NLP is a library for Natural Language Processing (NLP) in Python. " +
    "It's built with the very latest research in mind, and was designed from day one to support " +
    "rapid prototyping. PyTorch-NLP comes with pre-trained embeddings, samplers, dataset loaders," +
    " metrics, neural network modules and text encoders. It's open-source software, released " +
    "under the BSD3 license.",
    license='BSD',
    install_requires=['numpy', 'pandas', 'tqdm'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch nlp text',
    python_requires='>=3.5',

    # Package info
    packages=find_packages(exclude=['.vscode', 'build_tools', 'docs', 'tests']),
    zip_safe=True)

setup(**setup_info)
