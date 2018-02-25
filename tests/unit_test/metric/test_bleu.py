# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from lib.metrics.bleu import moses_multi_bleu
from lib.metrics import get_bleu
from lib.text_encoders import WordEncoder
from tests.lib.utils import get_batch


class TestMosesBleu(unittest.TestCase):
    """Tests using the Moses multi-bleu script to calculate BLEU score"""

    def _test_multi_bleu(self, hypotheses, references, lowercase, expected_bleu):
        """Runs a multi-bleu test."""
        result = moses_multi_bleu(hypotheses=hypotheses, references=references, lowercase=lowercase)
        np.testing.assert_almost_equal(result, expected_bleu, decimal=2)

    def test_multi_bleu(self):
        self._test_multi_bleu(
            hypotheses=np.array(
                ["The brown fox jumps over the dog 笑", "The brown fox jumps over the dog 2 笑"]),
            references=np.array([
                "The quick brown fox jumps over the lazy dog 笑",
                "The quick brown fox jumps over the lazy dog 笑"
            ]),
            lowercase=False,
            expected_bleu=46.51)

    def test_empty(self):
        self._test_multi_bleu(
            hypotheses=np.array([]), references=np.array([]), lowercase=False, expected_bleu=0.00)

    def test_multi_bleu_lowercase(self):
        self._test_multi_bleu(
            hypotheses=np.array(
                ["The brown fox jumps over The Dog 笑", "The brown fox jumps over The Dog 2 笑"]),
            references=np.array([
                "The quick brown fox jumps over the lazy dog 笑",
                "The quick brown fox jumps over the lazy dog 笑"
            ]),
            lowercase=True,
            expected_bleu=46.51)


class TestGetBleu(unittest.TestCase):

    def setUp(self):
        self.output_text_encoder = WordEncoder(['a b c d e'], append_eos=False)
        prediction = self.output_text_encoder.encode('a b c d d').tolist()
        target = self.output_text_encoder.encode('a b c d e').tolist()
        _, targets, outputs = get_batch(
            predictions=[prediction],
            targets=[target],
            vocab_size=self.output_text_encoder.vocab_size)
        self.targets = targets
        self.outputs = outputs

    def test_ignore_index_none(self):
        bleu = get_bleu(self.targets, self.outputs, self.output_text_encoder, print_=True)
        self.assertTrue(bleu >= 0.0)
        self.assertTrue(bleu <= 100.0)

    def test_ignore_index(self):
        bleu = get_bleu(
            self.targets,
            self.outputs,
            self.output_text_encoder,
            ignore_index=self.output_text_encoder.stoi['e'])
        self.assertTrue(bleu == 100.0)
