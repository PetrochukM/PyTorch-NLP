import os

from lib.checkpoint import Checkpoint


def test_checkpoint():
    # Test Sae
    directory = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = Checkpoint.save(directory, {'test': True})

    # Test Load
    checkpoint = Checkpoint(checkpoint_path)
    assert checkpoint.test

    # Test Recent
    checkpoint = Checkpoint.recent(directory)
    assert checkpoint.test

    # Clean up after test
    os.remove(checkpoint_path)

    # Test Recent
    checkpoint = Checkpoint.recent(directory)
    assert checkpoint is None
