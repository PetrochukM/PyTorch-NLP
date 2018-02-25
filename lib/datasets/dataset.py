from torch.utils import data

# TODO: Just use Pandas#DataFrame


class Dataset(data.Dataset):

    def __init__(self, rows):
        self.columns = set()
        for row in rows:
            self.columns.update(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self.columns:
                raise AttributeError
            return [row[key] for row in self.rows]
        else:
            return self.rows[key]

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        return key in self.columns
