""" Default reserved tokens

These tokens are used as global default parameters in this library; therefore, any method using
these will be consistent but also allow for customization.
"""
DEFAULT_PADDING_INDEX = 0
DEFAULT_UNKNOWN_INDEX = 1
DEFAULT_EOS_INDEX = 2
DEFAULT_SOS_INDEX = 3
DEFAULT_COPY_INDEX = 4
DEFAULT_PADDING_TOKEN = '<pad>'
DEFAULT_UNKNOWN_TOKEN = '<unk>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_SOS_TOKEN = '<s>'
DEFAULT_COPY_TOKEN = '<copy>'
DEFAULT_RESERVED_TOKENS = [
    DEFAULT_PADDING_TOKEN, DEFAULT_UNKNOWN_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_SOS_TOKEN,
    DEFAULT_COPY_TOKEN
]

DEFAULT_RESERVED_ITOS = DEFAULT_RESERVED_TOKENS
DEFAULT_RESERVED_STOI = {token: index for index, token in enumerate(DEFAULT_RESERVED_ITOS)}
