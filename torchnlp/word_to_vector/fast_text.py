# BSD 3-Clause License

# Copyright (c) James Bradbury and Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors


class FastText(_PretrainedWordVectors):
    """ Enriched word vectors with subword information from Facebook's AI Research (FAIR) lab.

    A approach based on the skipgram model, where each word is represented as a bag of character
    n-grams. A vector representation is associated to each character n-gram; words being
    represented as the sum of these representations.

    References:
        * https://arxiv.org/abs/1607.04606
        * https://fasttext.cc/
        * https://arxiv.org/abs/1710.04087

    Args:
        name (str or None, optional): The name of the file that contains the vectors
        url (str or None, optional): url for download if vectors not found in cache
        language (str): language of the vectors (only needed when both url and name 
            are ignored)
        aligned (bool): if True: use multilingual embeddings where words with
            the same meaning share (approximately) the same position in the
            vector space across languages. if False: use regular FastText
            embeddings. All available languages can be found under
            https://github.com/facebookresearch/MUSE#multilingual-word-embeddings.
            (only needed when both url and name are ignored)
        cache (str, optional): directory for cached vectors
        unk_init (callback, optional): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be any function that takes in a Tensor and
            returns a Tensor of the same size
        is_include (callable, optional): callable returns True if to include a token in memory
            vectors cache; some of these embedding files are gigantic so filtering it can cut
            down on the memory usage. We do not cache on disk if ``is_include`` is defined.

    Example:
        >>> from torchnlp.word_to_vector import FastText  # doctest: +SKIP
        >>> vectors = FastText()  # doctest: +SKIP
        >>> vectors['hello']  # doctest: +SKIP
        -0.1595
        -0.1826
        ...
        0.2492
        0.0654
        [torch.FloatTensor of size 300]
    """
    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'
    aligned_url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{}.align.vec'

    def __init__(self, language="en", url=None, name=None, aligned=False, **kwargs):
        if not name:
            if not url:
                if aligned:
                    url = self.aligned_url_base.format(language)
                else:
                    url = self.url_base.format(language)
            name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)
