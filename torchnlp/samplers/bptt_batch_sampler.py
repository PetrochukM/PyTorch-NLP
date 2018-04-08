import math

from torchnlp.samplers.bptt_sampler import BPTTSampler


class BPTTBatchSampler(object):
    """Samples sequentially a batch of source and target slices of size ``bptt_length``.

    Typically, such a sampler, is used for language modeling training with backpropagation through
    time (BPTT).

    **Reference:**
    https://github.com/pytorch/examples/blob/c66593f1699ece14a4a2f4d314f1afb03c6793d9/word_language_model/main.py#L61

    Args:
        data (iterable): Iterable data.
        bptt_length (int): Length of the slice.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        type_ (str, optional): Type of batch ['source'|'target'] to load where a target batch is one
            timestep ahead

    Example:
        >>> sampler = BPTTBatchSampler(range(100), bptt_length=2, batch_size=3, drop_last=False)
        >>> list(sampler)[0] # First Batch
        [slice(0, 2), slice(34, 36), slice(67, 69)]
    """

    def __init__(self, data, bptt_length, batch_size, drop_last, type_='source'):
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last

        # For each row in the batch, we iterate over a chunk of size `chunk_size`
        # Our chunks are similar to the columns in this PyTorch example:
        # https://github.com/pytorch/examples/blob/c66593f1699ece14a4a2f4d314f1afb03c6793d9/word_language_model/main.py#L61
        chunk_sizes = [math.floor(len(data) / batch_size)] * batch_size

        # Distribute the remaining elements to some chunks
        if not self.drop_last:
            remainder = len(data) - sum(chunk_sizes)
            for i in range(remainder):
                chunk_sizes[i] += 1

        self.samplers = [{
            'offset': sum(chunk_sizes[:i]),
            'sampler': BPTTSampler(range(chunk_sizes[i]), bptt_length, type_=type_)
        } for i in range(batch_size)]

    def __iter__(self):
        # Samplers iterate over chunks similar to:
        # https://github.com/pytorch/examples/blob/c66593f1699ece14a4a2f4d314f1afb03c6793d9/word_language_model/main.py#L112
        self.iterators = [iter(value['sampler']) for value in self.samplers]
        while True:
            batch = []
            for i, iterator in enumerate(self.iterators):
                try:
                    # Adjust the sampler indices to the offset
                    offset = self.samplers[i]['offset']
                    slice_ = next(iterator)
                    batch.append(slice(slice_.start + offset, slice_.stop + offset))
                except StopIteration:
                    pass

            # Samplers are all empty
            if (len(batch) == 0):
                break

            yield batch

    def __len__(self):
        return len(self.samplers[0]['sampler'])
