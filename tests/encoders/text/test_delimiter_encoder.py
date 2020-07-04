import pickle

import pytest

from torchnlp.encoders.text import DelimiterEncoder
from torchnlp.encoders.text import DEFAULT_UNKNOWN_TOKEN


@pytest.fixture
def encoder():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    return DelimiterEncoder('/', sample, append_eos=True, append_sos=True)


def test_delimiter_encoder(encoder):
    input_ = 'symbols/namesake/named_after'
    output = encoder.encode(input_)
    assert encoder.decode(output) == '/'.join(
        ['symbols', DEFAULT_UNKNOWN_TOKEN, DEFAULT_UNKNOWN_TOKEN])


def test_is_pickleable(encoder):
    pickle.dumps(encoder)
