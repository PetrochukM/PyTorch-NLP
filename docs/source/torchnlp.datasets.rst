torchnlp.datasets package
=========================

The ``torchnlp.datasets`` package introduces modules capable of downloading, caching and loading 
commonly used NLP datasets.

Modules return a :class:`torch.utils.data.Dataset` object i.e,
they have ``__getitem__`` and ``__len__`` methods implemented. Hence, they can all be passed to a
:class:`torch.utils.data.DataLoader` which can load multiple samples parallelly using
``torch.multiprocessing`` workers.

.. automodule:: torchnlp.datasets
    :members:
    :undoc-members:
