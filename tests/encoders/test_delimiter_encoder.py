from torchnlp.text_encoders import DelimiterEncoder
from torchnlp.text_encoders import UNKNOWN_TOKEN
from torchnlp.text_encoders import EOS_TOKEN


def test_delimiter_encoder():
    sample = ['people/deceased_person/place_of_death', 'symbols/name_source/namesakes']
    encoder = DelimiterEncoder('/', sample, append_eos=True)
    input_ = 'symbols/namesake/named_after'
    output = encoder.encode(input_)
    assert encoder.decode(output) == '/'.join(['symbols', UNKNOWN_TOKEN, UNKNOWN_TOKEN]) + EOS_TOKEN
