import os
import torch
import h5py
import pandas as pd
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import convolve1d

from data.datasets.librispeech import LibriDataset
from data.datasets.lorenz import LorenzDataset
from data.datasets.m1 import M1Dataset
from data.datasets.hc import HCDataset
from data.datasets.temp import TEMPDataset
from data.datasets.ms import MSDataset

def sum_over_chunks(X, stride):
    X_trunc = X[:len(X) - (len(X) % stride)]
    reshaped = X_trunc.reshape((len(X_trunc) // stride, stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed

def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd

def load_kording_paper_data(filename, bin_width_s=0.05, min_spike_count=10, preprocess=True):
    with open(filename, "rb") as fname:
        data = pickle.load(fname)
    X, Y = data[0], data[1]
    good_X_idx = (1 - (np.isnan(X[:, 0]) + np.isnan(X[:, 1]))).astype(np.bool)
    good_Y_idx = (1 - (np.isnan(Y[:, 0]) + np.isnan(Y[:, 1]))).astype(np.bool)
    good_idx = good_X_idx * good_Y_idx
    X, Y = X[good_idx], Y[good_idx]
    chunk_size = int(np.round(bin_width_s / 0.05))  # 50 ms default bin width
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size) / chunk_size
    X = X[:, np.sum(X, axis=0) > min_spike_count]
    if preprocess:
        X = np.sqrt(X)
        X = moving_center(X, n=600)
        Y -= Y.mean(axis=0, keepdims=True)
        Y /= Y.std(axis=0, keepdims=True)
    return {'neural': X, 'loc': Y}

def load_weather_data(filename):
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[['Vancouver', 'Portland', 'San Francisco', 'Seattle',
             'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque',
             'Denver', 'San Antonio', 'Dallas', 'Houston', 'Kansas City',
             'Minneapolis', 'Saint Louis', 'Chicago', 'Nashville', 'Indianapolis',
             'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte', 'Miami',
             'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal',
             'Boston']]
    df = df.dropna(axis=0, how='any')
    dts = (df.index[1:] - df.index[:-1]).to_numpy()
    df = df.iloc[np.nonzero(dts > dts.min())[0].max() + 1:]
    Xfs = df.values.copy()
    ds_factor = 24
    X = resample(Xfs, Xfs.shape[0] // ds_factor, axis=0)
    return X

def load_sabes_data(filename, bin_width_s=.05, preprocess=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            n_bins = int(np.floor((max_t - t[0]) / bin_width_s))
            binned_spikes = np.zeros((n_bins, d), dtype=np.int)
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times - t[0] < n_bins * bin_width_s]
                    bin_idx = np.floor((spike_times - t[0]) / bin_width_s).astype(np.int)
                    unique_idxs, counts = np.unique(bin_idx, return_counts=True)
                    # make sure to ignore the hash here...
                    binned_spikes[unique_idxs, chan_idx * n_sorted_units + unit_idx - 1] += counts
            binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > 0]
            if preprocess:
                binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > 5000]
                binned_spikes = np.sqrt(binned_spikes)
                binned_spikes = moving_center(binned_spikes, n=600)
            result[region] = binned_spikes
        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        # Line up the binned spikes with the cursor data
        t_mid_bin = np.arange(len(binned_spikes)) * bin_width_s + bin_width_s / 2
        cursor_pos_interp = interp1d(t - t[0], cursor_pos, axis=0)
        cursor_interp = cursor_pos_interp(t_mid_bin)
        if preprocess:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        result["cursor"] = cursor_interp
        return result

def load_accel_data(filename, preprocess=True):
    df = pd.read_csv(filename)
    X = df.values[:, 1:]
    if preprocess:
        X -= X.mean(axis=0, keepdims=True)
        X /= X.std(axis=0, keepdims=True)
    return X

def librispeech_loader(opt, num_workers=16):

    if opt.validate:
        print("Using Train / Val Split")
        train_dataset = LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/train_val_train.txt"
            ),
        )

        test_dataset = LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/train_val_val.txt"
            ),
        )

    else:
        print("Using Train+Val / Test Split")
        train_dataset = LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/train_split.txt"
            ),
        )

        test_dataset = LibriDataset(
            opt,
            os.path.join(
                opt.data_input_dir,
                "LibriSpeech/train-clean-100",
            ),
            os.path.join(
                opt.data_input_dir, "LibriSpeech100_labels_split/test_split.txt"
            ),
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset

def lorenz_loader(opt, num_workers=16, lorenz_length=100):
    # load data
    # import pdb; pdb.set_trace()
    with h5py.File(opt.data_input_dir, "r") as f:
        # snr_vals = f.attrs["snr_vals"][:]
        # X = f["X"][:]
        # X_dynamics = f["X_dynamics"][:]
        X_noisy_dset = f["X_noisy"][:]
        X_noisy = X_noisy_dset[opt.snr_index]


    if opt.validate:
        print("Using Train / Val Split")
        train_dataset = LorenzDataset(
            X_noisy, lorenz_length=lorenz_length
        )

        test_dataset = LorenzDataset(
            X_noisy, lorenz_length=lorenz_length
        )
    else:
        print("Using Train+Val / Test Split")
        train_dataset = LorenzDataset(
            X_noisy, lorenz_length=lorenz_length
        )

        test_dataset = LorenzDataset(
            X_noisy, lorenz_length=lorenz_length
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, train_dataset, test_loader, test_dataset

def m1_loader(opt, num_workers=16, length=100, good_ts=None):
    # load data

    M1 = load_sabes_data('/home/rui/Data/M1/indy_20160627_01.mat')
    X, Y = M1['M1'], M1['cursor']
    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    train_dataset = M1Dataset(
        X, length=length
    )

    test_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = None

    return train_loader, train_dataset, test_loader, test_dataset

def hc_loader(opt, num_workers=16, length=100, good_ts=22000):
    # load data

    HC = load_kording_paper_data('/home/rui/Data/HC/example_data_hc.pickle')
    X, Y = HC['neural'], HC['loc']
    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    train_dataset = HCDataset(
        X, length=length
    )

    test_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = None

    return train_loader, train_dataset, test_loader, test_dataset

def temp_loader(opt, num_workers=16, length=100, good_ts=None):
    # load data

    weather = load_weather_data('/home/rui/Data/TEMP/temperature.csv')
    X, Y = weather, weather
    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    train_dataset = TEMPDataset(
        X, length=length
    )

    test_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = None

    return train_loader, train_dataset, test_loader, test_dataset

def ms_loader(opt, num_workers=16, length=100, good_ts=None):
    # load data

    ms = load_accel_data('/home/rui/Data/motion_sense/A_DeviceMotion_data/std_6/sub_19.csv')
    X, Y = ms, ms
    if good_ts is not None:
        X = X[:good_ts]
        Y = Y[:good_ts]

    train_dataset = MSDataset(
        X, length=length
    )

    test_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = None

    return train_loader, train_dataset, test_loader, test_dataset

