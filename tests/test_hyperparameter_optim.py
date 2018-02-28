import unittest

from skopt.space import Integer

from torchnlp.utils import config_logging
from torchnlp.hyperparameter_optim import hyperband
from torchnlp.hyperparameter_optim import successive_halving

config_logging()

mock_dimensions = [Integer(1, 100, name='integer')]


def mock(resources, integer=0, checkpoint=None):
    # `integer` is a hyperparameter set the first batch
    if checkpoint is not None:
        return checkpoint, checkpoint
    return integer, integer


class TestHyperparameterOptimization(unittest.TestCase):

    def test_hyperband_simple(self):
        # Basic check on hyperband
        scores, hyperparameters = hyperband(objective=mock, dimensions=mock_dimensions)
        for score, hyperparameter in zip(scores, hyperparameters):
            self.assertEqual(score, hyperparameter['integer'])

    def test_successive_halving_simple(self):
        # Basic check on successive halving
        scores, hyperparameters = successive_halving(objective=mock, dimensions=mock_dimensions)
        for score, hyperparameter in zip(scores, hyperparameters):
            self.assertEqual(score, hyperparameter['integer'])

    def test_hyperband_no_progress_bar(self):
        # Basic check on hyperband
        scores, hyperparameters = hyperband(
            objective=mock, dimensions=mock_dimensions, progress_bar=False)
        for score, hyperparameter in zip(scores, hyperparameters):
            self.assertEqual(score, hyperparameter['integer'])

    def test_successive_halving_no_progress_bar(self):
        # Basic check on successive halving
        scores, hyperparameters = successive_halving(
            objective=mock, dimensions=mock_dimensions, progress_bar=False)
        for score, hyperparameter in zip(scores, hyperparameters):
            self.assertEqual(score, hyperparameter['integer'])

    def test_successive_halving_downsample(self):
        with self.assertRaises(ValueError):
            successive_halving(
                objective=mock,
                dimensions=mock_dimensions,
                progress_bar=False,
                downsample=1,
                n_models=45)


if __name__ == '__main__':
    unittest.main()
