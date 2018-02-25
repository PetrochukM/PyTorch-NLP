"""
Manages a global namespaced configuration.

TODO: Look into implementing configurable without decoraters:
  - Trace all function calls and intercept the call
  - Apply a decorator recursively too all modules in os.cwd()
    import pkgutil
    import os

    for loader, module_name, is_pkg in pkgutil.walk_packages([os.getcwd()]):
        print(module_name)

    http://code.activestate.com/recipes/577742-apply-decorators-to-all-functions-in-a-module/
"""
from functools import reduce
from collections import defaultdict

import ast
import inspect
import logging
import operator
import sys
import pprint
from importlib import import_module

import wrapt

pretty_printer = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__name__)


class _KeyListDictionary(dict):
    """
    Allows for lists of keys to query a deep dictionary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        """ Similar to dict.__getitem__ but allows key to be a list of keys """
        if isinstance(key, list):
            return reduce(operator.getitem, key, self)

        return super().__getitem__(key)

    def __contains__(self, key):
        """ Similar to dict.__contains__ but allows key to be a list of keys """
        if isinstance(key, list):
            pointer = self
            for k in key:
                if k in pointer:
                    pointer = pointer[k]
                else:
                    return False
            return True

        return super().__contains__(key)


# Private configuration for all modules in the Python repository
# DO NOT IMPORT. Use @configurable instead.
_configuration = _KeyListDictionary()


def _dict_merge(dict_, merge_dict, overwrite=False):
    """ Recursive `dict` merge. `dict_merge` recurses down into dicts nested to an arbitrary depth,
    updating keys. The `merge_dict` is merged into `dict_`.

    Args:
      dict_ (dict) dict onto which the merge is executed
      merge_dict (dict) dict merged into dict
    """
    for key in merge_dict:
        if key in dict_ and isinstance(dict_[key], dict):
            _dict_merge(dict_[key], merge_dict[key], overwrite)
        elif overwrite and key in dict_:
            dict_[key] = merge_dict[key]
        elif key not in dict_:
            dict_[key] = merge_dict[key]


def _parse_configuration(dict_):
    """
    Transform some `dict_` into a deep _KeyListDictionary that allows for module look ups.

    NOTE: interprets dict_ keys as python `dotted module names`.

    Example:
        `dict_`:
            {
              'abc.abc': {
                'cda': 'abc
              }
            }
        Returns:
            {
              'abc': {
                'abc': {
                  'cda': 'abc
                }
              }
            }
    """
    parsed = {}
    _parse_configuration_helper(dict_, parsed)
    return parsed


def _parse_configuration_helper(dict_, new_dict):
    """ Recursive helper to _parse_configuration """
    if not isinstance(dict_, dict):
        return

    for key in dict_:
        split = key.split('.')
        past_dict = new_dict
        for i, split_key in enumerate(split):
            if split_key == '':
                raise TypeError('Invalid config: Improper key format %s' % key)
            if i == len(split) - 1 and not isinstance(dict_[key], dict):
                if split_key in new_dict:
                    raise TypeError('Invalid config: Key %s already seen.' % key)
                new_dict[split_key] = dict_[key]
            else:
                if split_key not in new_dict:
                    new_dict[split_key] = {}
                new_dict = new_dict[split_key]
        _parse_configuration_helper(dict_[key], new_dict)
        new_dict = past_dict  # Reset dict


def _dict_to_flat_config(dict_):
    """
    Reduce a dictionary from deep to shallow by concatenating keys as python `dotted module names`.

    Example:
        `dict_`:
            {
              'abc.abc': {
                'cda': 'abc
              }
            }
        Returns:
            {
              'abc.abc.cda': 'abc'
            }
    Args:
        dict_ (dict)
    Returns:
        (dict) shallow dictionary with every key concatenated by "." similar to module names in
        python
    Raises:
        (TypeError) module names (keys) are formatted improperly (Example: 'lib..models')
        (TypeError) duplicate functions/modules/packages are defined
    """
    flat = {}
    parsed = _parse_configuration(dict_)  # Make sure it can be parsed
    _dict_to_flat_config_helper(parsed, flat, [])
    return flat


def _dict_to_flat_config_helper(dict_, flat_dict, keys):
    """ Recursive helper for `dict_to_flat_config` """
    for key in dict_:
        next_keys = keys + [key]
        if isinstance(dict_[key], dict):
            _dict_to_flat_config_helper(dict_[key], flat_dict, next_keys)
        else:
            flat_key = '.'.join(next_keys)
            if flat_key in flat_dict:
                raise TypeError('Invalid config: Key %s already seen.' % key)
            flat_dict[flat_key] = dict_[key]


def _check_configuration(dict_, keys=[]):
    """ Check the parsed configuration every module that it points too exists with @configurable.

    Cases to handle recursively:
        {
            'lib.nn': {
                'seq_encoder.SeqEncoder.__init__': {
                    'bidirectional': True,
                },
                'attention.Attention.__init__.attention_type': 'general',
            }
        }
    """
    if not isinstance(dict_, dict):
        # Recursive function walked up the chain and never found a @configurable
        logger.warn("""
Path %s does not contain @configurable.
NOTE: Due to Python remaining the __main__ module, this check can be ignored here.
NOTE: _check_configuration can be ignored for external libraries.
        """.strip(), keys)
        return

    if len(keys) >= 2:
        # Scenario: Function
        try:
            module_path = '.'.join(keys[:-1])
            module = import_module(module_path)
            if hasattr(module, keys[-1]):
                function = getattr(module, keys[-1])
                # `is True` to avoid truthy values
                if (hasattr(function, '__wrapped__')):  # TODO: Find a better check
                    return
        except (ImportError, AttributeError):
            pass

    if len(keys) >= 3:
        # Scenario: Class
        try:
            module_path = '.'.join(keys[:-2])
            module = import_module(module_path)
            if hasattr(module, keys[-2]):
                class_ = getattr(module, keys[-2])
                function = getattr(class_, keys[-1])
                if (hasattr(function, '__wrapped__')):
                    return
        except (ImportError, AttributeError):
            pass

    for key in dict_:
        _check_configuration(dict_[key], keys[:] + [key])


def add_config(dict_):
    """
    Add configuration to the global configuration.

    Example:
        `dict_`=
              {
                'lib': {
                  'models': {
                    'decoder_rnn.DecoderRNN.__init__': {
                      'embedding_size': 32
                      'rnn_size': 32
                      'n_layers': 1
                      'rnn_cell': 'gru'
                      'embedding_dropout': 0.0
                      'intra_layer_dropout': 0.0
                    }
                  }
                }
              }
    Args:
        dict_ (dict): configuration to add
        is_log (bool): Note the configuration log can be verbose. If false, do not log the added
            configuration.
    Returns: None
    Raises:
        (TypeError) module names (keys) are formatted improperly (Example: 'lib..models')
        (TypeError) duplicate functions/modules/packages are defined
    """
    global _configuration
    parsed = _parse_configuration(dict_)
    logger.info('Checking configuration...')
    _check_configuration(parsed)
    _dict_merge(_configuration, parsed, overwrite=True)
    _configuration = _KeyListDictionary(_configuration)
    logger.info('Configuration checked.')


def log_config():
    """
    Log the global configuration
    """
    logger.info('Global configuration:')
    logging.info(pretty_printer.pformat(_configuration))


def clear_config():
    """
    Clear the global configuration
    
    Returns: None
    """
    global _configuration
    _configuration = _KeyListDictionary()


def _get_module_name(func):
    """ Get the name of a module. Handles `__main__` by inspecting sys.argv[0]. """
    module = inspect.getmodule(func)
    if module.__name__ == '__main__':
        file_name = sys.argv[0]
        no_extension = file_name.split('.')[0]
        return no_extension.replace('/', '.')
    else:
        return module.__name__


@wrapt.decorator
def configurable(func, instance, args, kwargs):
    """
    Decorator peeks @ the global configuration that defines arguments for some functions. The
    arguments and key word arguments passed to the function are merged with the globally defined
    arguments.

    Args/Return are defined by `wrapt.decorator`.
    """
    global _configuration
    parameters = inspect.signature(func).parameters
    module_keys = _get_module_name(func).split('.')
    keys = module_keys + func.__qualname__.split('.')
    print_name = module_keys[-1] + '.' + func.__qualname__
    default = _configuration[keys] if keys in _configuration else {}  # Get default
    if not isinstance(default, dict):
        logger.info('%s:%s config malformed must be a dict of arguments', print_name,
                    '.'.join(keys))
    merged = default.copy()
    merged.update(kwargs)  # Add kwargs
    # Add args
    args = list(args)
    for parameter in parameters:
        if len(args) == 0 or parameters[parameter].kind == parameters[parameter].VAR_POSITIONAL:
            break
        merged[parameter] = args.pop(0)
        # No POSITIONAL_ONLY arguments
        # https://docs.python.org/3/library/inspect.html#inspect.Parameter
        assert parameter not in kwargs, "Python is broken. Args overwriting kwargs."

    try:
        if len(default) == 0:
            logger.info('%s no config for: %s', print_name, '.'.join(keys))
        # TODO: Does not print all parameters; FIX
        logger.info('%s was configured with:\n%s', print_name, pretty_printer.pformat(merged))
        return func(*args, **merged)
    except TypeError as error:
        logger.info('%s was passed defaults: %s', print_name, default)
        logger.error(error, exc_info=True)
        raise


class HyperparameterSpaceConfig(object):
    """
    Define a set of (key, value) pairs for a parameter space.

    NOTE: Each value represents a Real, Categorical or Integer dimension.
    NOTE: Keys define configurable modules in Python.
    NOTE: In order to allow sharing of dimensions, we used `id` to determine uniqueness.
    """

    def __init__(self, dict_):
        """
        Args:
          dict_ (dict): configuration where every configurable argument is a range of values.
        """
        self.space_config = _dict_to_flat_config(dict_)
        self.id_to_dimension = {}
        self.space_ids = sorted(
            [(id(value), key, value) for key, value in self.space_config.items()],
            key=lambda item: item[1])  # id changes, so we save it
        for id_, _, value in self.space_ids:
            if id_ in self.id_to_dimension:
                assert self.id_to_dimension[id_] is value
            else:
                self.id_to_dimension[id_] = value
        self.dimension_ids = list(self.id_to_dimension.keys())
        self.dimensions = list(self.id_to_dimension.values())
        logger.info('Got dimensions: %s', self.id_to_dimension)

    def get_hyperparameter_names(self, shorten_names=True):
        """
        NOTE: Size different from dimensions because some hyperparameters share dimensions

        Returns (list of str): configurable hyperparameter names
        """

        if shorten_names:
            splits = [name.split('.') for _, name, _ in self.space_ids]

            # Count words
            module_name_count = defaultdict(int)
            for split_path in splits:
                for module_name in split_path:
                    module_name_count[module_name] += 1

            def unique_name(split_path):
                sort = [(module_name_count[name], name) for name in split_path]
                sort = sorted(sort, key=lambda item: item[0], reverse=True)
                size = min(2, len(sort))
                return '.'.join([name for count, name in sort[-size:]])

            return [unique_name(split_path) for split_path in splits]
        return [key for _, key, _ in self.space_ids]

    def get_dimensions(self):
        """
        Get a list of dimensions with a space definition for each:
        [int, int] -> Range between two integers
        [float, float] -> Range between two floats
        [any] * 3+ -> Categorical dimension
        [int, int, str] -> Range between two integers with str defining a prior:
          "uniform" or "log-uniform"
        [float, float, str] -> Range between two floats with str defining a prior:
          "uniform" or "log-uniform"

        Returns: (list of lists): defines a list of ranges for each dimension.
        """
        return self.dimensions

    def config_to_point(self, config):
        """
        Given a config, transform it to a point.

        Args:
            config (dict) keys represent modules and value represent argument values
        Returns:
            (list): some points in the space of `self.dimensions`
        """
        flat_config = _dict_to_flat_config(config)
        id_to_dimension_value = {}
        for id_, key, _ in self.space_ids:
            assert key in flat_config, "%s is not defined in config" % key
            value = flat_config[key]
            if id_ in id_to_dimension_value:
                assert id_to_dimension_value[id_] is value, "%s must match %s" % (
                    value, id_to_dimension_value[id_])
            else:
                id_to_dimension_value[id_] = value
        return [id_to_dimension_value[id_] for id_ in self.dimension_ids]

    def point_to_config(self, point):
        """
        Given a point in dimensions, transform it back to a configuration in `space_config`.

        Args:
            point (list): some points in the space of `self.dimensions`
        Returns:
            (dict) keys represent modules and value represent argument values
        """
        named_point = self._unnamed_point_to_named(point)
        ret_config = {}
        for id_, key, value in self.space_ids:
            assert id_ in named_point, "%s:%s with id %d not found in point %s" % (key, value, id_,
                                                                                   named_point)
            ret_config[key] = named_point[id_]
        return ret_config

    def _unnamed_point_to_named(self, unnamed_point):
        """
        List to named dictionary.

        Motivation: https://scikit-optimize.github.io/optimizer/index.html#ask returns a list
            rather than a dictionary with dimension names.
        Args:
            unnamed_point (list of values)
        Returns:
            (dict)
        """
        named_point = {}
        assert len(self.dimension_ids) == len(
            unnamed_point), "%s point must have the same number of dimensions %d." % (
                unnamed_point, len(self.dimensions))
        for i, value in enumerate(unnamed_point):
            assert self.dimension_ids[i] not in named_point
            named_point[self.dimension_ids[i]] = value
        return named_point
