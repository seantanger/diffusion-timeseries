import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
from diffusion import DiffusionModel, SpikingDiffusionModel
from dataset import GBMDataset, HestonDataset
from metrics import plot_metrics_comparison, plot_paths_and_prices
# from syops import get_model_complexity_info
from scipy.fftpack import dct, idct
from torch.nn import DataParallel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False, help='train from scratch')
parser.add_argument('--folderdir', default='scale', type=str, help='folder path')
parser.add_argument('--schedule', default='linear', type=str, help='diffusion scheduler')
parser.add_argument('--parallel', default=False, help='parallel training')
parser.add_argument('--dataset', type=str, default='GBM', help='dataset name')
parser.add_argument('--spiking', default=False, help='spiking or non spiking diffusion model')

# Training
parser.add_argument('--resume', action='store_true', default=False, help="load pre-trained model")
parser.add_argument('--resume_model', type=str, help='resume model path')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='total epochs')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma')
parser.add_argument('--mu', type=float, default=0.05, help='drift')
parser.add_argument('--n_samples', type=int, default=10000, help='number of paths to generate')

args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
num_gpus = torch.cuda.device_count()
# print(f"Number of GPUS: {num_gpus}")
folder_path = args.folderdir
print(folder_path)
def train_diffusion_model(dataset, n_epochs, lr, batch_size, device, spiking):
    # Create dataset
    sequence_length = 63
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    if spiking:
        diffusion = SpikingDiffusionModel(n_steps=1000, sequence_length=sequence_length+1, device=device)
        print("Using Spiking Diffusion Model")
    else:
        diffusion = DiffusionModel(n_steps=1000, sequence_length=sequence_length+1, device=device)
        print("Using Non-Spiking Diffusion Model")
    diffusion.model.to(device)

    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'Training+Loading Resume model from {args.resume_model}')
        diffusion.model.load_state_dict(ckpt, strict=True)
    else:
        print('Training from scratch')

    if args.parallel:
        print("Running in parallel")
        diffusion.model = DataParallel(diffusion.model)

    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=lr)
    losses = []
    # Training loop
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss
            losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.4f}")
    return diffusion, losses

def main():
    # Parameters
    np.random.seed(0)
    if args.dataset=='GBM':
        S0 = 100       # Initial price
        K = 100        # Strike price
        T = 0.5          # Time to maturity (1 year)
        r = args.mu       # Risk-free rate
        sigma = args.sigma    # Volatility
        M = 100000    # Number of paths (large for convergence)
        N = 63 # sequence length
        t = np.linspace(0, T, N+1)  # Time array
        dt = T/N
        batch_size=args.batch_size
        print(f"mu={r}, sigma={sigma}, batch size={batch_size}, epochs={args.epochs}, T={T}")

        dataset = GBMDataset(n_samples=M, sequence_length=N, S0=S0, mu=r, sigma=sigma, T=T)
        paths = dataset.data
        paths = dataset.inverse_transform(paths)#.squeeze(1)
    elif args.dataset=='Heston':
        # Parameters
        S0 = 100       # Initial stock price
        v0 = 0.1**2      # Initial variance (volatility = sqrt(0.04) = 20%)
        mu = 0.05      # Drift
        kappa = 1    # Mean reversion speed
        theta = 0.1**2   # Long-run variance
        sigma = 0.1    # Volatility of volatility
        rho = .5     # Correlation between asset and vol shocks
        T = .5        # Time horizon (1 year)
        N = 63        # Number of time steps (daily)
        # M = 5          # Number of paths
        # Create dataset
        n_samples = 100000  # Number of sample paths
        seq_length = 63  # Number of time steps (daily)
        dataset = HestonDataset(n_samples, seq_length, S0, v0, mu, kappa, theta, sigma, rho, T)
        paths = dataset.data
        paths = dataset.inverse_transform(paths)  # Convert back to original scale



    # Train diffusion models
    if args.train:
        diffusion, losses = train_diffusion_model(dataset=dataset, n_epochs=args.epochs, lr=1e-5, batch_size=batch_size, device=device, spiking=args.spiking)
        torch.save(diffusion.model.state_dict(), f"./{folder_path}/unet_mu={r}_sigma={sigma}_t={T}.pt")
        print("model saved")
    else:
        diffusion = DiffusionModel(n_steps=1000, sequence_length=N+1, device=device)
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'Loading Resume model from {args.resume_model}')
        diffusion.model.load_state_dict(ckpt, strict=True)

    # Generate paths (Evaluate)
    generated_paths = diffusion.sample(n_samples = args.n_samples, batch_size = 500, device=device)
    generated_paths_transformed = dataset.inverse_transform(generated_paths)#.squeeze(1)
    np.savez(f'{folder_path}/paths.npz', matrix1=paths, matrix2=generated_paths_transformed)
    print(f"number of paths: {generated_paths_transformed.shape}")

    plot_paths_and_prices(paths, generated_paths_transformed, S0, K, T, r, N, sigma, folder_path)

    title = f'{folder_path}/metrics_mu={r}_sigma={sigma}_K={K}'
    diffusion_metrics = plot_metrics_comparison(real_paths = paths, generated_paths = generated_paths_transformed, title = title, dt=dt)

if __name__ == '__main__':
    main()
