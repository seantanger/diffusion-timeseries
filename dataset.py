import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats
from scipy.fftpack import dct, idct

class GBMDataset(Dataset):
    def __init__(self, n_samples, sequence_length, S0, mu, sigma, T=1):
        self.data = []
        # t = np.arange(sequence_length) * dt
        # print(t)
        dt = T / sequence_length
        t = np.linspace(0, T, sequence_length+1)
        dW = np.random.normal(0, np.sqrt(dt), size=(n_samples, sequence_length))
        W = np.cumsum(dW, axis=1)
        paths = S0 * np.exp((mu - 0.5 * sigma**2) * t[1:] + sigma * W)
        paths = np.hstack([S0 * np.ones((n_samples, 1)), paths])
        # paths = paths[:, :-1]
        self.data = np.array(paths)
        # Fit scaler on all data
        # self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = StandardScaler()

        flat_data = self.data.reshape(-1, 1)
        self.scaler.fit(flat_data)
        
        # Transform data
        scaled_data = self.scaler.transform(flat_data).reshape(n_samples, sequence_length+1)
        self.data = torch.tensor(scaled_data, dtype=torch.float32)
        self.data = self.data.unsqueeze(1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    def inverse_transform(self, scaled_data):
        """Convert scaled data back to original scale"""
        if isinstance(scaled_data, torch.Tensor):
            scaled_data = scaled_data.cpu().numpy()
        # Remove channel dimension if present
        if scaled_data.ndim == 3:
            scaled_data = scaled_data.squeeze(1)
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).reshape(scaled_data.shape)

class GBMDataset_DCT(Dataset):
    def __init__(self, n_samples, sequence_length, S0, mu, sigma, T=1):
        self.data = []
        self.sequence_length = sequence_length
        self.n_samples = n_samples

        # Generate GBM paths
        dt = T / sequence_length
        t = np.linspace(0, T, sequence_length + 1)
        dW = np.random.normal(0, np.sqrt(dt), size=(n_samples, sequence_length))
        W = np.cumsum(dW, axis=1)
        paths = S0 * np.exp((mu - 0.5 * sigma**2) * t[1:] + sigma * W)
        paths = np.hstack([S0 * np.ones((n_samples, 1)), paths])  # Add initial price S0
        self.data = np.array(paths)

        # Apply mirror reflection extension
        self.mirrored_paths = self.mirror_reflection(paths)

        # Apply Discrete Cosine Transform (DCT) to each mirrored path
        self.dct_data = np.apply_along_axis(dct, axis=1, arr=self.mirrored_paths, norm='ortho', type=2)

        # Scale DCT coefficients
        self.scaler = StandardScaler()
        flat_dct_data = self.dct_data.reshape(-1, 1)
        self.scaler.fit(flat_dct_data)
        self.scaled_dct_data = self.scaler.transform(flat_dct_data).reshape(self.dct_data.shape)

        # Convert to PyTorch tensor
        self.data = torch.tensor(self.scaled_dct_data, dtype=torch.float32)
        self.data = self.data.unsqueeze(1)  # Add channel dimension

    def mirror_reflection(self, paths):
        """
        Apply mirror reflection to each path to make it symmetric.
        For a path [x0, x1, ..., xn], the mirrored path is [x0, x1, ..., xn, xn, ..., x1, x0].
        """
        return np.hstack([paths, np.flip(paths, axis=1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def inverse_transform(self, scaled_dct_data):
        """Convert scaled DCT coefficients back to the original scale"""
        if isinstance(scaled_dct_data, torch.Tensor):
            scaled_dct_data = scaled_dct_data.cpu().numpy()

        # Remove channel dimension if present
        if scaled_dct_data.ndim == 3:
            scaled_dct_data = scaled_dct_data.squeeze(1)

        # Inverse scaling
        flat_dct_data = scaled_dct_data.reshape(-1, 1)
        unscaled_dct_data = self.scaler.inverse_transform(flat_dct_data).reshape(scaled_dct_data.shape)

        # Apply Inverse DCT
        mirrored_data = np.apply_along_axis(idct, axis=1, arr=unscaled_dct_data, norm='ortho', type=2)

        # Remove the mirrored part to recover the original path
        original_data = mirrored_data[:, :self.sequence_length + 1]
        return original_data