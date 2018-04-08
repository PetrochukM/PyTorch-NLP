import math

from torch.utils.data.sampler import Sampler


class BPTTSampler(Sampler):
    """Samples sequentially source and target slices of size ``bptt_length``.

    Typically, such a sampler, is used for language modeling training with backpropagation through
    time (BPTT).

    **Reference:**
    https://github.com/pytorch/examples/blob/c66593f1699ece14a4a2f4d314f1afb03c6793d9/word_language_model/main.py#L122

    Arguments:
        data (iterable): Iterable data.
        bptt_length (int): Length of the slice.
        type_ (str, optional): Type of slice ['source'|'target'] to load where a target slice is one
            timestep ahead

    Example:
        >>> from torchnlp.samplers import BPTTSampler
        >>> list(BPTTSampler(range(5), 2))
        [slice(0, 2), slice(2, 4)]
    """

    def __init__(self, data, bptt_length, type_='source'):
        self.data = data
        self.bptt_length = bptt_length
        self.type = type_

    def __iter__(self):
        for i in range(0, len(self.data) - 1, self.bptt_length):
            seq_length = min(self.bptt_length, len(self.data) - 1 - i)
            if self.type == 'source':
                yield slice(i, i + seq_length)
            if self.type == 'target':
                yield slice(i + 1, i + 1 + seq_length)

    def __len__(self):
        return math.ceil((len(self.data) - 1) / self.bptt_length)
