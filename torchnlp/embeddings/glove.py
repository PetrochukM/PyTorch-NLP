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

from torchnlp.embeddings.pretrained_embedding import _PretrainedEmbeddings


class GloVe(_PretrainedEmbeddings):
    """Word vectors derived from word-word co-occurrence statistics from a corpus by Stanford.

    GloVe is essentially a log-bilinear model with a weighted least-squares objective. The main
    intuition underlying the model is the simple observation that ratios of word-word co-occurrence
    probabilities have the potential for encoding some form of meaning.

    Reference Website:
    https://nlp.stanford.edu/projects/glove/

    Example:
        >>> from torchnlp.embeddings import GloVe
        >>> vectors = GloVe()
        >>> vectors['hello']
        -1.7494
        0.6242
        ...
        -0.6202
        2.0928
        [torch.FloatTensor of size 100]

    Args:
        name (str): name of the GloVe vectors ('840B', 'twitter.27B', '6B', '42B')
        cache (str, optional): directory for cached vectors
        unk_init (callback, optional): by default, initialize out-of-vocabulary word vectors
            to zero vectors; can be any function that takes in a Tensor and
            returns a Tensor of the same size
        is_include (callable, optional): callable returns True if to include a token in memory
            vectors cache; some of these embedding files are gigantic so filtering it can cut
            down on the memory usage. We do not cache on disk if `is_include` is defined.
    """

    url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)
