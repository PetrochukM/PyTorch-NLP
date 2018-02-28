"""
We implement additional hyperparameter optimization methods not present in
https://scikit-optimize.github.io/.

Gist: https://gist.github.com/Deepblue129/2c5fae9daf0529ed589018c6353c9f7b
"""

import math
import logging
import random

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _random_points(dimensions, n_points, random_seed=None):
    """ Generate a random sample of points from dimensions """
    # NOTE: We supply as `randint` to `random_state`; otherwise, dimensions with the same distribution would
    # recive the same sequence of random numbers.
    # We seed `random` so the random seeds generated are deterministic.
    random.seed(random_seed)
    points = {
        d.name: d.rvs(n_samples=n_points, random_state=random.randint(0, 2**32))
        for d in dimensions
    }
    points = [{k: points[k][i] for k in points} for i in range(n_points)]
    return points


def successive_halving(
        objective,
        dimensions,
        max_resources_per_model=81,
        downsample=3,  # Test random downsamples work and check boundaries
        initial_resources=3,
        n_models=45,
        random_seed=None,
        progress_bar=True):
    """
    Adaptation of the Successive Halving algorithm.

    tl;dr keep the best models every iteration of the `initial_models` downsampling each time

    Adaptation: Instead of running for N / 2 models for 2T, we run N / 2 models for T**downsample.
    This adaptation is the same adaptation of Successive Halving in hyperband.

    Reference: http://proceedings.mlr.press/v51/jamieson16.pdf
    Reference: http://www.argmin.net/2016/06/23/hyperband/

    TODO: Splitting in half is a fairly random number. We could possibly look for a better split
    point. For example, we could use the large margin to split points. Or predict the performance
    of hyperparameters would do well in the future.

    Args:
        objective (callable): objective function to minimize
            Named Args:
                resources (int): number of resources (e.g. epochs) to use while training model
                checkpoint (any): saved data from past run
                **hyperparameters (any): hyperparameters to run
            Returns:
                score (float): score to minimize
                checkpoint (any): saved data from run
        dimensions (list of skopt.Dimensions): list of dimensions to minimize under
        max_resources_per_model: Max number of resources (e.g. epochs) to use per model
        downsample: Downsampling of models (e.g. halving is a downsampling of 2)
        initial_resources: Number of resources (e.g. epochs) to use initially to evaluate first
          round.
        n_models (int): Number of models to evaluate
        random_seed (int, optional): Random seed for generating hyperparameters
        progress_bar (boolean or tqdm): Iff to use or update a progress bar.
    Returns:
        scores (list of floats): Scores of the top objective executions
        hyperparameters (list of lists of dict): Hyperparameters with a one to one correspondence
            to scores.
    """
    if downsample <= 1:
        raise ValueError('Downsample must be > 1; otherwise, the number of resources allocated' +
                         'does not grow')

    round_n_models = lambda n: max(round(n), 1)

    total_resources_per_model = 0
    hyperparameters = _random_points(dimensions, round_n_models(n_models), random_seed)
    checkpoints = [None for _ in range(round_n_models(n_models))]
    scores = [math.inf for _ in range(round_n_models(n_models))]

    # Create a new progress bar
    remember_to_close = False
    if not isinstance(progress_bar, tqdm) and progress_bar:
        remember_to_close = True
        # TODO: Compute the tqdm total
        progress_bar = tqdm()
        # Keep tabs on a set of stats
        setattr(progress_bar, 'stats', {'min_score': math.inf, 'models_evaluated': 0})

    while total_resources_per_model < max_resources_per_model:
        # Compute number of resources to continue running each model with
        if total_resources_per_model == 0:
            update_n_resources = initial_resources
        else:
            update_n_resources = min(
                total_resources_per_model * downsample - total_resources_per_model,
                max_resources_per_model - total_resources_per_model)

        results = []
        for score, checkpoint, params in zip(scores, checkpoints, hyperparameters):
            new_score, new_checkpoint = objective(
                resources=update_n_resources, checkpoint=checkpoint, **params)
            new_score = min(score, new_score)
            results.append(tuple([new_score, new_checkpoint]))
            if isinstance(progress_bar, tqdm):
                progress_bar.update(update_n_resources)
                if progress_bar.stats['min_score'] > new_score:
                    progress_bar.stats['min_score'] = new_score
                    progress_bar.set_postfix(progress_bar.stats)

        total_resources_per_model += update_n_resources

        # NOTE: If this is not the last
        is_last_iteration = total_resources_per_model >= max_resources_per_model
        if not is_last_iteration:
            # Sort by minimum score `k[0][0]`
            results = sorted(zip(results, hyperparameters), key=lambda k: k[0][0])
            models_evaluated = len(results) - round_n_models(n_models / downsample)
            results = results[:round_n_models(n_models / downsample)]
            # Update `hyperparameters` lists
            results, hyperparameters = zip(*results)
            n_models = n_models / downsample
        else:
            models_evaluated = len(results)

        # Update `scores` and `checkpoints` lists
        scores, checkpoints = zip(*results)

        if isinstance(progress_bar, tqdm):
            progress_bar.stats['models_evaluated'] += models_evaluated
            progress_bar.set_postfix(progress_bar.stats)

    if remember_to_close:
        progress_bar.close()

    return scores, hyperparameters


def hyperband(objective,
              dimensions,
              max_resources_per_model=81,
              downsample=3,
              total_resources=None,
              random_seed=None,
              progress_bar=True):
    """
    Adaptation of the Hyperband algorithm

    tl;dr search over the space of successive halving hyperparameters

    Adaptation: Originally Hyperband was implemented with the assumption that we cannot reuse
    models. We redid the math allowing for reusing models. This is particularly helpful in speeding
    up 1 GPU hyperparameter optimization. Just to clarify, by reusing models, we mean that
    given hyperparameters `x` and epochs `y`, we can use one model to evaluate all `y` integers
    with hyperparameters `x`.

    Reference: https://arxiv.org/pdf/1603.06560.pdf
    Reference: http://www.argmin.net/2016/06/23/hyperband/

    TODO: Implement extension to hyperband proporting an increase of x4:
    https://arxiv.org/pdf/1705.10823.pdf
    http://www.ijcai.org/Proceedings/15/Papers/487.pdf

    Args:
        objective (callable): objective function to minimize
            Named Args:
                resources (int): number of resources (e.g. epochs) to use while training model
                checkpoint (any): saved data from past run
                **hyperparameters (any): hyperparameters to run
            Returns:
                score (float): score to minimize
                checkpoint (any): saved data from run
        dimensions (list of skopt.Dimensions): list of dimensions to minimize under
        max_resources_per_model (float): Max number of resources (e.g. epochs) to use per model
        downsample (int): Downsampling of models (e.g. halving is a downsampling of 2)
        total_resources (optional): Max number of resources hyperband is allowed to use over the
            entirety of the algorithm.
        random_seed (int, optional): Random seed for generating hyperparameters
        progress_bar (boolean, optional): Boolean for displaying tqdm
    Returns:
        scores (list of floats): Scores of the top objective executions
        hyperparameters (list of lists of dict): Hyperparameters with a one to one correspondence
            to scores.
    """
    if downsample <= 1:
        raise ValueError('Downsample must be > 1; otherwise, the number of resources allocated' +
                         'does not grow')

    all_scores = []
    all_hyperparameters = []

    # Number of times to run hyperband
    # Ex. `max_resources_per_model = 81 and downsample = 3`
    #     Then => initial_resources = [1, 3, 9, 27, 81]
    #     And => `hyperband_rounds = 5`
    #     And => `successive_halving_rounds = [5, 4, 3, 2, 1]`
    n_hyperband_rounds = math.floor(math.log(max_resources_per_model, downsample)) + 1
    if total_resources is None:
        # TODO: Multiply by the number of dimensions so it scales the number of models
        # given the large space
        total_resources_per_round = max_resources_per_model * n_hyperband_rounds
    else:
        total_resources_per_round = total_resources / n_hyperband_rounds
    total_models_evaluated = 0

    if progress_bar:
        progress_bar = tqdm(total=total_resources_per_round * n_hyperband_rounds)
        setattr(progress_bar, 'stats', {'min_score': math.inf, 'models_evaluated': 0})

    for i in reversed(range(n_hyperband_rounds)):
        n_successive_halving_rounds = i + 1

        # NOTE: Attained by running the below code on https://sandbox.open.wolframcloud.com:
        #   Reduce[Power[d, j - 1] * (x / Power[d, j]) +
        #   Sum[(Power[d, i] - Power[d, i - 1]) * (x / Power[d, i]), {i, j, k}] == e
        #   && k >=j>=1 && k>=1 && d>=1, {x}]
        # `e` is `total_resources_per_round`
        # `x` is `n_models`
        # `k - j` is `i`
        # `d` is downsample
        # The summation is similar to the successive halving rounds loop. It computes the number
        # of resources with reuse run in total. This is different from hyperband that assumes
        # no reuse.
        n_models = downsample * total_resources_per_round
        n_models /= downsample * (1 + i) - i
        n_models /= downsample**(-i + n_hyperband_rounds - 1)
        total_models_evaluated += n_models

        scores, hyperparameters = successive_halving(
            objective=objective,
            dimensions=dimensions,
            max_resources_per_model=max_resources_per_model,
            downsample=downsample,
            initial_resources=max_resources_per_model / downsample**i,
            n_models=n_models,
            random_seed=random_seed,
            progress_bar=progress_bar)
        logger.info('Finished hyperband round: %d of %d', n_hyperband_rounds - i - 1,
                    n_hyperband_rounds - 1)
        all_scores.extend(scores)
        all_hyperparameters.extend(hyperparameters)

    if isinstance(progress_bar, tqdm):
        progress_bar.close()

    logger.info('Total models evaluated: %f', total_models_evaluated)
    logger.info('Total resources used: %f', total_resources_per_round * n_hyperband_rounds)
    logger.info('Total resources used per model on average: %f',
                total_models_evaluated / total_resources_per_round * n_hyperband_rounds)

    return all_scores, all_hyperparameters
