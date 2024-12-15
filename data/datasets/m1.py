import numpy as np
from torch.utils.data import Dataset


class M1Dataset(Dataset):
    def __init__(
        self,
        data,
        length=None,
        mean=None,
        std=None
    ):
        self.data = data
        if length is None:
            self.length = self.data.shape[0]
        else:
            self.length = length
        num_ts = self.data.shape[0] - self.length + 1
        if mean is None:
            self.mean = np.mean(self.data, axis=0)
        else:
            self.mean = mean
        if std is None:
            self.std = np.std(self.data, axis=0)
        else:
            self.std = std
        self.data_standardized = (self.data - self.mean)/self.std
        self.segmented_data = np.stack([self.data_standardized[i: i+self.length].T for i in range(num_ts)])

    def __getitem__(self, index):
        return self.segmented_data[index]

    def __len__(self):
        return self.segmented_data.shape[0]

    def get_full_data(self):
        return self.data_standardized.T