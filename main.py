import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats
from diffusion import DiffusionModel, SpikingDiffusionModel, train_diffusion_model
from metrics import plot_metrics_comparison, plot_paths_and_prices, run_multiple_mc
# from syops import get_model_complexity_info
from scipy.fftpack import dct, idct
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folderdir', default='scale', type=str, help='folder path')
parser.add_argument('--schedule', default='linear', type=str, help='diffusion scheduler')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder_path = args.folderdir # or "./dct_scale/", "./scale_dct/"

    
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

def plot_timestep_losses(timestep_losses, title):
    timesteps = sorted(timestep_losses.keys())
    avg_losses = [timestep_losses[t] for t in timesteps]

    # Plot
    plt.plot(timesteps, avg_losses)
    plt.title("Average loss per timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f'{folder_path}/{title}.png')
    plt.close()


if __name__ == '__main__':
    # Parameters
    S0 = 100       # Initial price
    K = 105        # Strike price
    T = 1          # Time to maturity (1 year)
    r = 0.02       # Risk-free rate
    sigma = 0.1    # Volatility
    M = 10000    # Number of paths (large for convergence)
    N = 63 # sequence length
    t = np.linspace(0, T, N+1)  # Time array
    batch_size=16
    sbatch_size=8

    dataset = GBMDataset(n_samples=M, sequence_length=N, S0=S0, mu=r, sigma=sigma, T=T)
    paths = dataset.data
    paths = dataset.inverse_transform(paths)#.squeeze(1)


    # Train diffusion models
    diffusion, losses = train_diffusion_model(dataset=dataset, n_epochs=1000, lr=1e-5, batch_size=batch_size, device=device, spiking=False)
    torch.save(diffusion.model.state_dict(), f"./parameters/timeseries_unet_mu={r}_sigma={sigma}_t={T}")
    spiking_diffusion, spiking_losses = train_diffusion_model(dataset=dataset, n_epochs=1000, lr=1e-5, batch_size=sbatch_size, device=device, spiking=True)
    torch.save(spiking_diffusion.model.state_dict(), f"./parameters/spiking_timeseries_unet_mu={r}_sigma={sigma}_t={T}")

    # # Plot timestep losses
    # plot_timestep_losses(timestep_losses=timestep_losses, title='timestep_losses')
    # plot_timestep_losses(timestep_losses=spiking_timestep_losses, title='spiking_timestep_losses')

    # Generate paths
    generated_paths = diffusion.sample(n_samples = M, batch_size = 500, device=device)
    generated_paths_transformed = dataset.inverse_transform(generated_paths)#.squeeze(1)
    spiking_generated_paths = spiking_diffusion.sample(n_samples = M, batch_size = 500, device=device)
    spiking_generated_paths_transformed = dataset.inverse_transform(spiking_generated_paths)#.squeeze(1)

    plot_paths_and_prices(paths, generated_paths_transformed, spiking_generated_paths_transformed, S0, K, T, r, N, sigma, folder_path)

    title = f'{folder_path}/metrics_mu={r}_sigma={sigma}_K={K}'
    diffusion_metrics = plot_metrics_comparison(real_paths = paths, generated_paths = generated_paths_transformed, title = title)

    title = f'{folder_path}/spiking_metrics_mu={r}_sigma={sigma}_K={K}'
    spiking_metrics = plot_metrics_comparison(real_paths = paths, generated_paths = spiking_generated_paths_transformed, title = title)
    # run_multiple_mc(dataset, diffusion, spiking_diffusion, S0, K, T, r, N, sigma, 3, device, folder_path)