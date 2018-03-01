from __future__ import unicode_literals

import array
import io
import logging
import os
import zipfile

from six.moves.urllib.request import urlretrieve
from tqdm import tqdm

import six
import tarfile
import torch

from torchnlp.utils import reporthook

logger = logging.getLogger(__name__)

# TODO: Credit torchtext for this snippet


class _PretrainedEmbeddings(object):

    def __init__(self,
                 name,
                 cache='.pretrained_embeddings_cache',
                 url=None,
                 unk_init=torch.Tensor.zero_,
                 is_include=None):
        """
        Args:
            name: name of the file that contains the vectors
            cache: directory for cached vectors
            url: url for download if vectors not found in cache
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size
            is_include (callable): callable returns True if to include a token in memory vectors
                cache; some of these embedding files are gigantic so filtering it can cut
                down on the memory usage. We do not cache on disk if is_include is defined.

        """
        self.unk_init = unk_init
        self.is_include = is_include
        self.name = name
        self.cache(name, cache, url=url)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def __len__(self):
        return len(self.vectors)

    def __str__(self):
        return self.name

    def cache(self, name, cache, url=None):
        if os.path.isfile(name):
            path = name
            path_pt = os.path.join(cache, os.path.basename(name)) + '.pt'
        else:
            path = os.path.join(cache, name)
            path_pt = path + '.pt'

        print(path)

        if not os.path.isfile(path_pt) or self.is_include is not None:
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                print(dest)
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        urlretrieve(url, dest, reporthook=reporthook(t))
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                print(ext)
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    with tarfile.open(dest, 'r:gz') as tar:
                        tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            # str call is necessary for Python 2/3 compatibility, since
            # argument must be Python 2 str (Python 3 bytes) or
            # Python 3 str (Python 2 unicode)
            itos, vectors, dim = [], array.array(str('d')), None

            # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]
            # If there are malformed lines, read in binary mode
            # and manually decode each word from utf-8
            except:
                logger.warning("Could not read {} as UTF8 file, "
                               "reading file as bytes and skipping "
                               "words with malformed UTF8.".format(path))
                with open(path, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            logger.info("Loading vectors from {}".format(path))
            for line in tqdm(lines, total=len(lines)):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(b" " if binary_lines else " ")

                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError("Vector for token {} has {} dimensions, but previously "
                                       "read vectors have {} dimensions. All vectors must have "
                                       "the same number of dimensions.".format(
                                           word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                if self.is_include is not None and not self.is_include(word):
                    continue

                vectors.extend(float(x) for x in entries)
                itos.append(word)

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)


class GloVe(_PretrainedEmbeddings):
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


class FastText(_PretrainedEmbeddings):

    url_base = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.vec'

    def __init__(self, language="en", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


class CharNGram(_PretrainedEmbeddings):

    name = 'charNgram.txt'
    url = ('http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/'
           'jmt_pre-trained_embeddings.tar.gz')

    def __init__(self, **kwargs):
        super(CharNGram, self).__init__(self.name, url=self.url, **kwargs)

    def __getitem__(self, token):
        vector = torch.Tensor(self.dim).zero_()
        if token == "<unk>":
            return self.unk_init(vector)
        # These literals need to be coerced to unicode for Python 2 compatibility
        # when we try to join them with read ngrams from the files.
        chars = ['#BEGIN#'] + list(token) + ['#END#']
        num_vectors = 0
        for n in [2, 3, 4]:
            end = len(chars) - n + 1
            grams = [chars[i:(i + n)] for i in range(end)]
            for gram in grams:
                gram_key = '{}gram-{}'.format(n, ''.join(gram))
                if gram_key in self.stoi:
                    vector += self.vectors[self.stoi[gram_key]]
                    num_vectors += 1
        if num_vectors > 0:
            vector /= num_vectors
        else:
            vector = self.unk_init(vector)
        return vector
