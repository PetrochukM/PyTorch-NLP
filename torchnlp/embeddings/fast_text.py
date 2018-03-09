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

from torchnlp.embeddings.pretrained_embedding import _PretrainedEmbeddings


class FastText(_PretrainedEmbeddings):
    """ Enriched word vectors with subword information from Facebook's AI Research (FAIR) lab.

    A approach based on the skipgram model, where each word is represented as a bag of character
    n-grams. A vector representation is associated to each character n-gram; words being
    represented as the sum of these representations.

    Paper Reference:
    https://arxiv.org/abs/1607.04606

    Main Website:
    https://fasttext.cc/

    Example:
        >>> from torchnlp.embeddings import FastText
        >>> vectors = FastText()
        >>> vectors['hello']
        -1.7494
        0.6242
        ...
        -0.6202
        2.0928
        [torch.FloatTensor of size 100]

    Args:
        language (str): language of the vectors
        cache (str, optional): directory for cached vectors
        unk_init (callback, optional): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be any function that takes in a Tensor and
            returns a Tensor of the same size
        is_include (callable, optional): callable returns True if to include a token in memory
            vectors cache; some of these embedding files are gigantic so filtering it can cut
            down on the memory usage. We do not cache on disk if `is_include` is defined.
    """
    url_base = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.vec'

    def __init__(self, language="en", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)
