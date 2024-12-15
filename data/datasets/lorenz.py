import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class LorenzDataset(Dataset):
    def __init__(
        self,
        data,
        lorenz_length=None
    ):
        self.data = data
        if lorenz_length is None:
            self.lorenz_length = self.data.shape[0]
        else:
            self.lorenz_length = lorenz_length
        num_ts = self.data.shape[0] - self.lorenz_length + 1
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.data_standardized = (self.data - self.mean)/self.std
        self.segmented_data = np.stack([self.data_standardized[i: i+self.lorenz_length].T for i in range(num_ts)])

    def __getitem__(self, index):
        return self.segmented_data[index]

    def __len__(self):
        return self.segmented_data.shape[0]

    def get_full_data(self):
        return self.data_standardized.T