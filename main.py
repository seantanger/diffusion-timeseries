	
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# import scipy.stats as stats
# from syops import get_model_complexity_info


device = torch.device('cuda')

class GBMDataset(Dataset):
    def __init__(self, n_samples, sequence_length, S0=100, mu=0.1, sigma=0.2, T=1):
        self.data = []
        # t = np.arange(sequence_length) * dt
        # print(t)
        dt = T / sequence_length
        t = np.linspace(0, T, sequence_length+1)
        dW = np.random.normal(0, np.sqrt(dt), size=(n_samples, sequence_length))
        W = np.cumsum(dW, axis=1)
        paths = S0 * np.exp((mu - 0.5 * sigma**2) * t[1:] + sigma * W)
        paths = np.hstack([S0 * np.ones((n_samples, 1)), paths])
        paths = paths[:, :-1]
        self.data = np.array(paths)
        # Fit scaler on all data
        self.scaler = StandardScaler()
        flat_data = self.data.reshape(-1, 1)
        self.scaler.fit(flat_data)
        
        # Transform data
        scaled_data = self.scaler.transform(flat_data).reshape(n_samples, sequence_length)
        self.data = torch.tensor(scaled_data, dtype=torch.float32)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data = self.data.unsqueeze(1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    def inverse_transform(self, scaled_data):
        """Convert scaled data back to original scale"""
        if isinstance(scaled_data, torch.Tensor):
            scaled_data = scaled_data.cpu().numpy()
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).reshape(scaled_data.shape)
