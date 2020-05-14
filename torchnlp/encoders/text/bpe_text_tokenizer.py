import codecs
from typing import List, Any, Tuple
from subword_nmt import learn_bpe, apply_bpe
from collections import Counter
from sacremoses import MosesTokenizer, MosesDetokenizer


class BPETextTokenizer(object):
    _moses_tok = MosesTokenizer(lang='en')
    _moses_detok = MosesDetokenizer(lang='en')

    def __init__(self, file_prefix=None, separator='@@'):
        if file_prefix is not None:
            self.codes_file = '{}.vocab'.format(file_prefix)

        self.separator = separator
        self.bpe = None
        self.vocab = None

    @staticmethod
    def pre_tokenize(line):
        return BPETextTokenizer._moses_tok.tokenize(line, return_str=True)

    @staticmethod
    def _segment_words(line, pre_apply=None):
        if pre_apply is not None:
            line = pre_apply(line)
        line = str(line)
        return line.strip('\r\n ').split()

    @staticmethod
    def get_vocabulary(item_list, segment=_segment_words, from_filenames=True):
        vocab = Counter()
        if from_filenames:
            for fname in item_list:
                with codecs.open(fname, encoding='UTF-8') as f:
                    for line in f:
                        for word in segment(line):
                            vocab[word] += 1
        else:
            for line in item_list:
                for word in segment(line):
                    vocab[word] += 1
        return vocab

    def build_from_corpus(self, item_list, min_count=2, num_symbols=10000,
                          total_symbols=False, from_filenames=True):
        def segment_words(line): return self._segment_words(line, self.pre_tokenize)

        vocab_words = self.get_vocabulary(item_list, segment_words, from_filenames=from_filenames)

        vocab_list = ['{0} {1}'.format(key, freq)
                      for (key, freq) in vocab_words.items()]

        with codecs.open(self.codes_file, 'w', encoding='UTF-8') as output:
            learn_bpe.learn_bpe(vocab_list, output, num_symbols=num_symbols,
                                min_frequency=min_count, verbose=False,
                                is_dict=True, total_symbols=total_symbols)

        with codecs.open(self.codes_file, encoding='UTF-8') as codes:
            self.bpe = apply_bpe.BPE(codes, separator=self.separator)

        self.vocab = dict(self.get_vocabulary(item_list=item_list, segment=self.segment,
                                              from_filenames=from_filenames))

    def segment(self, line):
        if not hasattr(self, 'bpe'):
            raise NameError('Learn bpe first!')
        line = self.pre_tokenize(line)
        return self.bpe.segment(line.strip('\r\n ')).split(' ')

    def encode(self, raw_text):
        return self.segment(raw_text)

    def decode(self, bpe_text, delimiter=' '):
        decode_string = delimiter.join(bpe_text)
        try:
            decode_string = decode_string.decode('utf-8')
        except Exception:
            pass
        decode_string = decode_string \
            .replace(self.separator + ' ', '') \
            .replace(self.separator, '')
        decode_string = str(decode_string).strip('\r\n ').split()
        decode_string = self._moses_detok.tokenize(decode_string)
        return decode_string
