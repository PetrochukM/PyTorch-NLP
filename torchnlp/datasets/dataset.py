from torch.utils import data


class Dataset(data.Dataset):
    """ A class implementing :class:`torch.utils.data.Dataset`.

    Dataset subclasses the abstract class :class:`torch.utils.data.Dataset`. The class overrides
    ``__len__``, ``__getitem__``, ``__contains__`` and ``__init__``.

    Dataset is a two-dimensional size-mutable, potentially heterogeneous tabular data structure
    with labeled axes (rows and columns).
    """

    def __init__(self, rows):
        """
        Args:
            rows (list of dict): Construct a two-dimensional tabular data structure from rows/
        """
        self.columns = set()
        for row in rows:
            self.columns.update(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        """
        Get a column or row from the dataset.

        Args:
            key (str or int): String referencing a column or integer referencing a row
        Returns:
            :class:`list` or :class:`dict`: List of column values or a dict representing a row
        """
        # Given an column string return list of column values.
        if isinstance(key, str):
            if key not in self.columns:
                raise AttributeError
            return [row[key] if key in row else None for row in self.rows]
        # Given an row integer return a object of row values.
        else:
            return self.rows[key]

    def __len__(self):
        """ Return the number of rows. """
        return len(self.rows)

    def __contains__(self, key):
        """ Check if dataset contains column. """
        return key in self.columns
