# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import logging
import re
import sys
import unicodedata

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

logger = logging.getLogger(__name__)

# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i)
    for i in xrange(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))

# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")
_ESCAPE_CHARS = set(u"\\_u;0123456789")


def native_to_unicode_py2(s):
    """Python 2: transform native string to Unicode."""
    return s if isinstance(s, unicode) else s.decode("utf8")  # noqa: F821


# Conversion between Unicode and UTF-8, if required (on Python2)
if six.PY2:
    native_to_unicode = native_to_unicode_py2
    unicode_to_native = lambda s: s.encode("utf-8")
else:
    # No conversion required on Python3
    native_to_unicode = lambda s: s
    unicode_to_native = lambda s: s


def encode(text):
    """
    Encode a unicode string as a list of tokens.
    Args:
      text: a unicode string
    Returns:
      a list of tokens as Unicode strings
    """
    if not text:
        return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    for pos in xrange(1, len(text)):
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[token_start:pos]
            if token != u" " or token_start == 0:
                ret.append(token)
            token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    return ret


def decode(tokens):
    """
    Decode a list of tokens to a unicode string.
    Args:
      tokens: a list of Unicode strings
    Returns:
      a unicode string
    """
    token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
    ret = []
    for i, token in enumerate(tokens):
        if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
            ret.append(u" ")
        ret.append(token)
    return "".join(ret)


def _escape_token(token, alphabet):
    """
    Escape away underscores and OOV characters and append '_'.
    This allows the token to be experessed as the concatenation of a list
    of subtokens from the vocabulary. The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.
    Args:
      token: A unicode string to be escaped.
      alphabet: A set of all characters in the vocabulary's alphabet.
    Returns:
      escaped_token: An escaped unicode string.
    Raises:
      ValueError: If the provided token is not unicode.
    """
    if not isinstance(token, six.text_type):
        raise ValueError("Expected string type for token, got %s" % type(token))

    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
    return u"".join(ret) + "_"


def _unescape_token(escaped_token):
    """
    Inverse of _escape_token().
    Args:
      escaped_token: a unicode string
    Returns:
      token: a unicode string
    """

    def match(m):
        if m.group(1) is None:
            return u"_" if m.group(0) == u"\\u" else u"\\"

        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return ""

    trimmed = escaped_token[:-1] if escaped_token.endswith("_") else escaped_token
    return _UNESCAPE_REGEX.sub(match, trimmed)


class SubwordTextTokenizer(object):
    """ Class for invertibly encoding text using a limited vocabulary.

    Invertibly encodes a native string as a sequence of subtokens from a limited
    vocabulary.

    A SubwordTextTokenizer is built from a corpus (so it is tailored to the text in
    the corpus), and stored to a file. See text_encoder_build_subword.py.
    It can then be loaded and used to encode/decode any text.

    Encoding has four phases:
        1.  Tokenize into a list of tokens.  Each token is a unicode string of either
            all alphanumeric characters or all non-alphanumeric characters.  We drop
            tokens consisting of a single space that are between two alphanumeric
            tokens.
        2.  Escape each token.  This escapes away special and out-of-vocabulary
            characters, and makes sure that each token ends with an underscore, and
            has no other underscores.
        3.  Represent each escaped token as a the concatenation of a list of subtokens
            from the limited vocabulary.  Subtoken selection is done greedily from
            beginning to end.  That is, we construct the list in order, always picking
            the longest subtoken in our vocabulary that matches a prefix of the
            remaining portion of the encoded token.
        4.  Concatenate these lists.  This concatenation is invertible due to the
            fact that the trailing underscores indicate when one list is finished.
    """

    def __init__(self):
        """Initialize and read from a file, if provided."""
        self._alphabet = set()

    def encode(self, raw_text):
        """Converts a native string to a list of subtoken.

        Args:
          raw_text: a native string.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        return self._tokens_to_subtoken(encode(native_to_unicode(raw_text)))

    def decode(self, subtokens):
        """Converts a sequence of subtoken to a native string.

        Args:
          subtokens: a list of integers in the range [0, vocab_size)
        Returns:
          a native string
        """
        return unicode_to_native(decode(self._subtoken_to_tokens(subtokens)))

    @property
    def vocab(self):
        return self._all_subtoken_strings

    @property
    def vocab_size(self):
        return len(self._all_subtoken_strings)

    def _tokens_to_subtoken(self, tokens):
        """ Converts a list of tokens to a list of subtoken.

        Args:
          tokens: a list of strings.
        Returns:
          a list of integers in the range [0, vocab_size)
        """
        ret = []
        for token in tokens:
            ret.extend(
                self._escaped_token_to_subtoken_strings(_escape_token(token, self._alphabet)))
        return ret

    def _subtoken_to_tokens(self, subtokens):
        """ Converts a list of subtoken to a list of tokens.

        Args:
          subtokens: a list of integers in the range [0, vocab_size)
        Returns:
          a list of strings.
        """
        concatenated = "".join(subtokens)
        split = concatenated.split("_")
        return [_unescape_token(t + "_") for t in split if t]

    def _escaped_token_to_subtoken_strings(self, escaped_token):
        """ Converts an escaped token string to a list of subtoken strings.

        Args:
          escaped_token: An escaped token as a unicode string.
        Returns:
          A list of subtokens as unicode strings.
        """
        # NOTE: This algorithm is greedy; it won't necessarily produce the "best"
        # list of subtokens.
        ret = []
        start = 0
        token_len = len(escaped_token)
        while start < token_len:
            for end in xrange(min(token_len, start + self._max_subtoken_len), start, -1):
                subtoken = escaped_token[start:end]
                if subtoken in self._all_subtoken_strings:
                    ret.append(subtoken)
                    start = end
                    break

            else:  # Did not break
                # If there is no possible encoding of the escaped token then one of the
                # characters in the token is not in the alphabet. This should be
                # impossible and would be indicative of a bug.
                assert False, "Token substring not found in subtoken vocabulary."

        return ret

    @classmethod
    def _count_tokens(cls, *sources):
        token_counts = collections.Counter()
        for corpus in sources:
            for text in corpus:
                token_counts.update(encode(text))
        return token_counts

    @classmethod
    def build_to_target_size_from_corpus(cls,
                                         *corpuses,
                                         target_size=32000,
                                         min_val=1,
                                         max_val=1e3,
                                         num_iterations=4):
        token_counts = SubwordTextTokenizer._count_tokens(*corpuses)
        return SubwordTextTokenizer.build_to_target_size_from_token_counts(
            target_size, token_counts, min_val, max_val, num_iterations)

    @classmethod
    def build_to_target_size_from_token_counts(cls,
                                               target_size,
                                               token_counts,
                                               min_val,
                                               max_val,
                                               num_iterations=4):
        """Builds a SubwordTextTokenizer that has `vocab_size` near `target_size`.

        Uses simple recursive binary search to find a minimum token count that most
        closely matches the `target_size`.

        Args:
          target_size: Desired vocab_size to approximate.
          token_counts: A dictionary of token counts, mapping string to int.
          min_val: An integer; lower bound for the minimum token count.
          max_val: An integer; upper bound for the minimum token count.
          num_iterations: An integer; how many iterations of refinement.

        Returns:
          A SubwordTextTokenizer instance.

        Raises:
          ValueError: If `min_val` is greater than `max_val`.
        """
        if min_val > max_val:
            raise ValueError("Lower bound for the minimum token count "
                             "is greater than the upper bound.")

        def bisect(min_val, max_val):
            """Bisection to find the right size."""
            present_count = (max_val + min_val) // 2
            logger.info("Trying min_count %d" % present_count)
            subtokenizer = cls()
            subtokenizer.build_from_token_counts(token_counts, present_count, num_iterations)
            logger.info("min_count %d attained a %d vocab_size", present_count,
                        subtokenizer.vocab_size)

            # If min_val == max_val, we can't do any better than this.
            if subtokenizer.vocab_size == target_size or min_val >= max_val:
                return subtokenizer

            if subtokenizer.vocab_size > target_size:
                other_subtokenizer = bisect(present_count + 1, max_val)
            else:
                other_subtokenizer = bisect(min_val, present_count - 1)

            if other_subtokenizer is None:
                return subtokenizer

            if (abs(other_subtokenizer.vocab_size - target_size) <
                    abs(subtokenizer.vocab_size - target_size)):
                return other_subtokenizer
            return subtokenizer

        return bisect(min_val, max_val)

    def build_from_corpus(self, *corpuses, min_count=1, num_iterations=4):
        token_counts = SubwordTextTokenizer._count_tokens(*corpuses)
        return self.build_from_token_counts(token_counts, min_count, num_iterations)

    def build_from_token_counts(self, token_counts, min_count, num_iterations=4):
        """Train a SubwordTextTokenizer based on a dictionary of word counts.

        Args:
          token_counts: a dictionary of Unicode strings to int.
          min_count: an integer - discard subtokens with lower counts.
          num_iterations: an integer; how many iterations of refinement.
        """
        self._init_alphabet_from_tokens(six.iterkeys(token_counts))

        # Bootstrap the initial list of subtokens with the characters from the
        # alphabet plus the escaping characters.
        self._init_subtokens_from_list(list(self._alphabet))

        # We build iteratively.  On each iteration, we segment all the words,
        # then count the resulting potential subtokens, keeping the ones
        # with high enough counts for our new vocabulary.
        if min_count < 1:
            min_count = 1
        for i in xrange(num_iterations):

            # Collect all substrings of the encoded token that break along current
            # subtoken boundaries.
            subtoken_counts = collections.defaultdict(int)
            for token, count in six.iteritems(token_counts):
                escaped_token = _escape_token(token, self._alphabet)
                subtokens = self._escaped_token_to_subtoken_strings(escaped_token)
                start = 0
                for subtoken in subtokens:
                    for end in xrange(start + 1, len(escaped_token) + 1):
                        new_subtoken = escaped_token[start:end]
                        subtoken_counts[new_subtoken] += count
                    start += len(subtoken)

            # Array of sets of candidate subtoken strings, by length.
            len_to_subtoken_strings = []
            for subtoken_string, count in six.iteritems(subtoken_counts):
                lsub = len(subtoken_string)
                if count >= min_count:
                    while len(len_to_subtoken_strings) <= lsub:
                        len_to_subtoken_strings.append(set())
                    len_to_subtoken_strings[lsub].add(subtoken_string)

            # Consider the candidates longest to shortest, so that if we accept
            # a longer subtoken string, we can decrement the counts of its
            # prefixes.
            new_subtoken_strings = []
            for lsub in xrange(len(len_to_subtoken_strings) - 1, 0, -1):
                subtoken_strings = len_to_subtoken_strings[lsub]
                for subtoken_string in subtoken_strings:
                    count = subtoken_counts[subtoken_string]
                    if count >= min_count:
                        # Exclude alphabet tokens here, as they must be included later,
                        # explicitly, regardless of count.
                        if subtoken_string not in self._alphabet:
                            new_subtoken_strings.append((count, subtoken_string))
                        for l in xrange(1, lsub):
                            subtoken_counts[subtoken_string[:l]] -= count

            # Include the alphabet explicitly to guarantee all strings are
            # encodable.
            new_subtoken_strings.extend((subtoken_counts.get(a, 0), a) for a in self._alphabet)
            new_subtoken_strings.sort(reverse=True)

            # Reinitialize to the candidate vocabulary.
            self._init_subtokens_from_list([subtoken for _, subtoken in new_subtoken_strings])

    def _init_subtokens_from_list(self, subtoken_strings):
        """Initialize token information from a list of subtoken strings."""
        # we remember the maximum length of any subtoken to avoid having to
        # check arbitrarily long strings.
        self._all_subtoken_strings = set([s for s in subtoken_strings if s])
        self._max_subtoken_len = max([len(s) for s in subtoken_strings])

    def _init_alphabet_from_tokens(self, tokens):
        """Initialize alphabet from an iterable of token or subtoken strings."""
        # Include all characters from all tokens in the alphabet to guarantee that
        # any token can be encoded. Additionally, include all escaping
        # characters.
        self._alphabet = {c for token in tokens for c in token}
        self._alphabet |= _ESCAPE_CHARS
