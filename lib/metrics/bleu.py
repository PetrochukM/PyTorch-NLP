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

import os
import re
import subprocess
import tempfile
import logging

import numpy as np

from six.moves import urllib
from six.moves import zip

logger = logging.getLogger(__name__)


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
      hypotheses: A numpy array of strings where each string is a single example.
      references: A numpy array of strings where each string is a single example.
      lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
      The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")
        os.chmod(multi_bleu_path, 0o755)
    except:
        logger.info("Unable to fetch multi-bleu.perl script, using local.")
        metrics_dir = os.path.dirname(os.path.realpath(__file__))
        bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", "..", "bin"))
        multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                logger.warning("multi-bleu.perl script returned non-zero exit code")
                logger.warning(error.output)
            bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return np.float32(bleu_score)


def get_bleu(targets, outputs, output_text_encoder, ignore_index=None, print_=False):
    """ Compute BLEU with standard moses multi-bleu.perl """
    decoded_targets = []
    decoded_predictions = []
    for target, output in zip(targets, outputs):
        target = target.squeeze(dim=0)
        output = output.squeeze(dim=0)
        prediction = output.max(1)[1].view(-1)
        if ignore_index is not None:
            mask = target.ne(ignore_index)
            target = target.masked_select(mask)
            prediction = prediction.masked_select(mask)
        decoded_predictions.append(output_text_encoder.decode(prediction))
        decoded_targets.append(output_text_encoder.decode(target))
    if decoded_targets == 0:
        bleu = None
    else:
        bleu = moses_multi_bleu(np.array(decoded_predictions), np.array(decoded_targets))
    if print_:
        logger.info('BLEU: %s [%d Corpus Size]', bleu, len(decoded_targets))
    return bleu
