import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.stats as stats
from diffusion import DiffusionModel, SpikingDiffusionModel
from dataset import GBMDataset
from metrics import plot_metrics_comparison, plot_paths_and_prices
# from syops import get_model_complexity_info
from scipy.fftpack import dct, idct
import argparse
from torch.nn import DataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False, help='train from scratch')
parser.add_argument('--folderdir', default='results', type=str, help='folder path')
parser.add_argument('--schedule', default='linear', type=str, help='diffusion scheduler')
parser.add_argument('--parallel', default=False, help='parallel training')
# Training
parser.add_argument('--resume', action='store_true', default=False, help="load pre-trained model")
parser.add_argument('--resume_model', type=str, help='resume model path')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='total epochs')
parser.add_argument('--sigma', type=float, default=0.1, help='sigma')
parser.add_argument('--mu', type=float, default=0.05, help='drift')
parser.add_argument('--n_samples', type=int, default=10000, help='number of paths to gnerate')

args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
num_gpus = torch.cuda.device_count()
print(f"Number of GPUS: {num_gpus}")
folder_path = args.folderdir
print(folder_path)

def train_diffusion_model(dataset, n_epochs, lr, batch_size, device, spiking):
    # Create dataset
    sequence_length = 63
    # dataset = GBMDataset(10000, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    if spiking:
        diffusion = SpikingDiffusionModel(n_steps=1000, sequence_length=sequence_length+1, device=device)
    else:
        diffusion = DiffusionModel(n_steps=1000, sequence_length=sequence_length+1, device=device)
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
    filter=False
    print(f"mu={r}, sigma={sigma}, batch size={batch_size}, epochs={args.epochs}, T={T}")
    dataset = GBMDataset(n_samples=M, sequence_length=N, S0=S0, mu=r, sigma=sigma, T=T)
    paths = dataset.data
    paths = dataset.inverse_transform(paths)#.squeeze(1)

    # Train diffusion models
    # diffusion, losses = train_diffusion_model(dataset=dataset, n_epochs=1000, lr=5e-5, batch_size=batch_size, device=device, spiking=False)
    # torch.save(diffusion.model.state_dict(), f"./parameters/timeseries_unet_mu={r}_sigma={sigma}_t={T}")
    if args.train:
        spiking_diffusion, spiking_losses = train_diffusion_model(dataset=dataset, n_epochs=args.epochs, lr=1e-5, batch_size=batch_size, device=device, spiking=True)
        torch.save(spiking_diffusion.model.state_dict(), f"./{folder_path}/spiking_unet_mu={r}_sigma={sigma}_t={T}.pt")
        print(f"spiking model saved")
    else:
        spiking_diffusion = SpikingDiffusionModel(n_steps=1000, sequence_length=N+1, device=device)
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'Loading Resume model from {args.resume_model}')
        spiking_diffusion.model.load_state_dict(ckpt, strict=True)

    # # Plot timestep losses
    # plot_timestep_losses(timestep_losses=timestep_losses, title='timestep_losses')
    # plot_timestep_losses(timestep_losses=spiking_timestep_losses, title='spiking_timestep_losses')

    # Generate paths
    spiking_generated_paths = spiking_diffusion.sample(n_samples = args.n_samples, batch_size = 500, device=device)
    spiking_generated_paths_transformed = dataset.inverse_transform(spiking_generated_paths)
    if filter:
        spiking_generated_paths_transformed = spiking_generated_paths_transformed[(spiking_generated_paths_transformed > 0).all(axis=1)]
    np.savez(f'{folder_path}/paths.npz', matrix1=paths, matrix2=spiking_generated_paths_transformed)
    print(f"number of paths: {spiking_generated_paths_transformed.shape}")

    plot_paths_and_prices(paths, spiking_generated_paths_transformed, S0, K, T, r, N, sigma, folder_path)

    title = f'{folder_path}/metrics_mu={r}_sigma={sigma}_K={K}'
    diffusion_metrics = plot_metrics_comparison(real_paths = paths, generated_paths = spiking_generated_paths_transformed, title = title, dt=dt)


if __name__ == '__main__':
    main()