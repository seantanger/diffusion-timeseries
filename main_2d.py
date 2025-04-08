import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats
from diffusion_2d import DiffusionModel_2D, train_2d_spiking_diffusion_model
# from metrics import plot_metrics_comparison, plot_paths_and_prices, run_multiple_mc
from metrics import monte_carlo_option_price, black_scholes_price
# from syops import get_model_complexity_info
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

        # Fit scaler on all data
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        flat_data = self.data.reshape(-1, 1)
        self.scaler.fit(flat_data)

        # Transform data
        scaled_data = self.scaler.transform(flat_data).reshape(n_samples, sequence_length + 1)

        # Reshape data into images of shape (n_images, 1, 128, 128)
        self.data = self._reshape_to_images(scaled_data)

    def _reshape_to_images(self, data):
        """
        Reshape the data into images of shape (n_images, 1, 128, 128).
        Each image represents 128 paths of length 128.
        """
        n_samples, sequence_length = data.shape

        # Ensure the sequence length is 128
        if sequence_length != (self.sequence_length+1):
            raise ValueError("Sequence length must be 128 to reshape into 128x128 images.")

        # Ensure the number of samples is a multiple of 128
        if n_samples % (self.sequence_length+1) != 0:
            raise ValueError("Number of samples must be a multiple of 128 to reshape into 128x128 images.")

        # Reshape into (n_images, 128, 128)
        n_images = n_samples // (self.sequence_length+1)
        images = data.reshape(n_images, (self.sequence_length+1), (self.sequence_length+1))

        # Add channel dimension
        images = images[:, np.newaxis, :, :]

        # Convert to PyTorch tensor
        return torch.tensor(images, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def inverse_transform(self, scaled_data):
        """Convert scaled data back to original scale"""
        if isinstance(scaled_data, torch.Tensor):
            scaled_data = scaled_data.cpu().numpy()

        # Remove channel dimension if present
        if scaled_data.ndim == 4:
            scaled_data = scaled_data.squeeze(1)

        # Reshape back to (n_samples, sequence_length)
        n_images, height, width = scaled_data.shape
        scaled_data = scaled_data.reshape(n_images * height, width)

        # Inverse scaling
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).reshape(scaled_data.shape)
    
    
def plot_paths_and_prices(original_paths, spiking_diffusion_paths, S0, K, T, r, N, sigma, folder_path):
    # Compute call option prices at each time step
    call_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="call")
    call_prices_spiking_diffusion = monte_carlo_option_price(spiking_diffusion_paths, K, T, r, N, option_type="call")
    call_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="call")

    # Compute put option prices at each time step
    put_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="put")
    put_prices_spiking_diffusion = monte_carlo_option_price(spiking_diffusion_paths, K, T, r, N, option_type="put")
    put_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="put")

    # Plot: Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    t = np.linspace(0, T, N+1)  # Time array
    # Plot call option prices
    ax1.plot(t, call_prices_mc, label="Monte Carlo Call Price", color="blue", linestyle="--")
    ax1.plot(t, call_prices_spiking_diffusion, label="Monte Carlo Call Price - spiking diffusion", color="green", linestyle="--")
    ax1.plot(t, call_prices_bs, label="Black-Scholes Call Price", color="red", linestyle="-")
    ax1.set_title("European Call Option Price at Each Time Step")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Option Price")
    # ax1.set_ylim(0, 10)
    ax1.legend()
    ax1.grid(True)

    # Plot put option prices
    ax2.plot(t, put_prices_mc, label="Monte Carlo Put Price", color="blue", linestyle="--")
    ax2.plot(t, put_prices_spiking_diffusion, label="Monte Carlo Put Price - spiking diffusion", color="green", linestyle="--")
    ax2.plot(t, put_prices_bs, label="Black-Scholes Put Price", color="red", linestyle="-")
    ax2.set_title("European Put Option Price at Each Time Step")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Option Price")
    # ax2.set_ylim(0, 10)
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()
    # Save the combined plot
    plt.savefig(f'{folder_path}/combined_prices_mu={r}_sigma={sigma}_K={K}.png')
    # Close the figure to free up memory
    plt.close()

    # Compare the 3 methods: Create a figure with three subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    # Calculate global y-axis limits
    y_min = min(
        np.min(original_paths),
        np.min(spiking_diffusion_paths))
    y_max = max(
        np.max(original_paths),
        np.max(spiking_diffusion_paths))
    # Plot original GBM paths
    ax1.plot(t, original_paths.T, lw=1)  # Transpose paths to (M, N)
    ax1.set_title("Original GBM Paths")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Price")
    ax1.set_ylim(y_min, y_max)  # Set y-axis limits
    ax1.grid(True)

    # Plot spiking diffusion-generated paths
    ax2.plot(t, spiking_diffusion_paths.T, lw=1)  # Transpose paths to (M, N)
    ax2.set_title("Spiking Diffusion-Generated Paths")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Price")
    ax2.set_ylim(y_min, y_max)  # Set y-axis limits
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{folder_path}/generated_paths_mu={r}_sigma={sigma}_K={K}.png')

    # Close the figure to free up memory
    plt.close()

def save_to_folder(images, folder_path):
    """
    Save the dataset to a folder as .npy files.
    Each file contains a batch of 32x32 grayscale images.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Convert data to numpy array
    data_np = images.numpy()  # Shape: (n_images, 1, 32, 32)

    # Save each image as a separate .npy file
    for i, image in enumerate(data_np):
        file_path = os.path.join(folder_path, f"image_{i:04d}.npy")
        np.save(file_path, image)

    print(f"Saved {len(data_np)} images to {folder_path}")

if __name__ == '__main__':
    # Parameters
    S0 = 100       # Initial price
    K = 105        # Strike price
    T = 1          # Time to maturity (1 year)
    r = 0.05       # Risk-free rate
    sigma = 0    # Volatility
    M = 12800    # Number of paths (large for convergence)
    N = 31 # sequence length
    t = np.linspace(0, T, N+1)  # Time array
    batch_size=16

    dataset = GBMDataset(n_samples=M, sequence_length=N, S0=S0, mu=r, sigma=sigma, T=T)
    original_paths = dataset.inverse_transform(dataset.data)  # Shape: (M, 128)
    print(f'path shape: {original_paths.shape}')

    diffusion, losses = train_2d_spiking_diffusion_model(dataset=dataset, n_epochs=10000, lr=1e-5, batch_size=batch_size, device=device)

    # Generate paths
    generated_paths = diffusion.sample(n_samples = 100, batch_size = 32, device=device)
    generated_paths_transformed = dataset.inverse_transform(generated_paths)#.squeeze(1)
    print(f'generated path shape: {generated_paths_transformed.shape}')
    plot_paths_and_prices(original_paths, generated_paths_transformed, S0, K, T, r, N, sigma, folder_path)

    generated_images = (generated_paths + 1) / 2
    generated_images = generated_images.cpu()
    save_to_folder(images=generated_images, folder_path="gbm_dataset")

    # title = f'{folder_path}/metrics_mu={r}_sigma={sigma}_K={K}'
    # diffusion_metrics = plot_metrics_comparison(real_paths = paths, generated_paths = generated_paths_transformed, title = title)

    # title = f'{folder_path}/spiking_metrics_mu={r}_sigma={sigma}_K={K}'
    # spiking_metrics = plot_metrics_comparison(real_paths = paths, generated_paths = spiking_generated_paths_transformed, title = title)