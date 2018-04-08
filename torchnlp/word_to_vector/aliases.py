from functools import partial

from torchnlp.word_to_vector.char_n_gram import CharNGram
from torchnlp.word_to_vector.fast_text import FastText
from torchnlp.word_to_vector.glove import GloVe

aliases = {
    "charngram.100d": partial(CharNGram),
    "fasttext.en.300d": partial(FastText, language="en"),
    "fasttext.simple.300d": partial(FastText, language="simple"),
    "glove.42B.300d": partial(GloVe, name="42B", dim="300"),
    "glove.840B.300d": partial(GloVe, name="840B", dim="300"),
    "glove.twitter.27B.25d": partial(GloVe, name="twitter.27B", dim="25"),
    "glove.twitter.27B.50d": partial(GloVe, name="twitter.27B", dim="50"),
    "glove.twitter.27B.100d": partial(GloVe, name="twitter.27B", dim="100"),
    "glove.twitter.27B.200d": partial(GloVe, name="twitter.27B", dim="200"),
    "glove.6B.50d": partial(GloVe, name="6B", dim="50"),
    "glove.6B.100d": partial(GloVe, name="6B", dim="100"),
    "glove.6B.200d": partial(GloVe, name="6B", dim="200"),
    "glove.6B.300d": partial(GloVe, name="6B", dim="300")
}
