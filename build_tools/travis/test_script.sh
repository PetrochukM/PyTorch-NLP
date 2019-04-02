#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# Exit immediately if a command exits with a non-zero status.
set -e

export PYTHONPATH=.

python --version

if [[ "$RUN_DOCS" == "true" ]]; then
    make -C docs html
fi

if [[ "$RUN_FLAKE8" == "true" ]]; then
    flake8 torchnlp/
    flake8 tests/
fi

run_tests() {
    TEST_CMD="python -m pytest tests/ torchnlp/ --verbose --durations=20 --cov=torchnlp --doctest-modules"
    if [[ "$RUN_SLOW" == "true" ]]; then
        TEST_CMD="$TEST_CMD --runslow"
    fi
    $TEST_CMD
}

run_tests
