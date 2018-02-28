from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, rows):
        self.columns = set()
        for row in rows:
            self.columns.update(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        # Given an column string return list of column values.
        if isinstance(key, str):
            if key not in self.columns:
                raise AttributeError
            return [row[key] if key in row else None for row in self.rows]
        # Given an row integer return a object of row values.
        else:
            return self.rows[key]

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        # Check if `self.columns` contains column
        return key in self.columns
