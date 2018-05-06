from numpy.testing import assert_almost_equal

import numpy
import torch
import unittest

from torchnlp.nn import CNNEncoder

# from allennlp.nn import InitializerApplicator


class TestCNNEncoder(unittest.TestCase):

    def test_get_dimension_is_correct(self):
        encoder = CNNEncoder(embedding_dim=5, num_filters=4, ngram_filter_sizes=(3, 5))
        assert encoder.get_output_dim() == 8
        assert encoder.get_input_dim() == 5
        encoder = CNNEncoder(
            embedding_dim=5, num_filters=4, ngram_filter_sizes=(3, 5), output_dim=7)
        assert encoder.get_output_dim() == 7
        assert encoder.get_input_dim() == 5

    def test_forward_does_correct_computation(self):
        encoder = CNNEncoder(embedding_dim=2, num_filters=1, ngram_filter_sizes=(1, 2))
        for param in encoder.parameters():
            torch.nn.init.constant_(param, 1.)
        input_tensor = torch.FloatTensor([[[.7, .8], [.1, 1.5]]])
        encoder_output = encoder(input_tensor)
        assert_almost_equal(
            encoder_output.data.numpy(), numpy.asarray([[1.6 + 1.0, 3.1 + 1.0]]), decimal=6)

    def test_forward_runs_with_larger_input(self):
        encoder = CNNEncoder(
            embedding_dim=7, num_filters=13, ngram_filter_sizes=(1, 2, 3, 4, 5), output_dim=30)
        tensor = torch.rand(4, 8, 7)
        assert encoder(tensor).size() == (4, 30)
