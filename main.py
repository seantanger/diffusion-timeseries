	
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from diffusion import DiffusionModel, SpikingDiffusionModel
# from syops import get_model_complexity_info


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # paths = paths[:, :-1]
        self.data = np.array(paths)
        # Fit scaler on all data
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
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1)).reshape(scaled_data.shape)
    
def monte_carlo_option_price(paths, K, T, r, option_type="call"):
    """
    Compute the price of a European option using Monte Carlo simulation.

    Parameters:
    S0         : Initial price
    K          : Strike price
    T          : Time to maturity
    r          : Risk-free rate
    sigma      : Volatility
    M          : Number of paths
    option_type: "call" or "put"

    Returns:
    option_price : Estimated option price
    """
    # Simulate GBM paths
    # paths = simulate_gbm(S0, r, sigma, T, N=252, M=M)

    # Get the terminal prices (S_T)
    S_T = paths[:, -1]

    # Compute payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs

    # Compute option price as the average of discounted payoffs
    option_price = np.mean(discounted_payoffs)

    return option_price

def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    """
    Compute the Black-Scholes price of a European option.

    Parameters:
    S0         : Initial price
    K          : Strike price
    T          : Time to maturity
    r          : Risk-free rate
    sigma      : Volatility
    option_type: "call" or "put"

    Returns:
    option_price : Black-Scholes option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        option_price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == "put":
        option_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return option_price

def train_diffusion_model(n_epochs=100, batch_size=64, device="cuda"):
    # Create dataset
    sequence_length = 127
    dataset = GBMDataset(10000, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    diffusion = DiffusionModel(sequence_length=sequence_length+1, device=device)
    diffusion.model.to(device)
    optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=1e-4)
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

# Parameters
S0 = 100       # Initial price
K = 105        # Strike price
T = 1          # Time to maturity (1 year)
r = 0.02       # Risk-free rate
sigma = 0.2    # Volatility
M = 10000    # Number of paths (large for convergence)
N = 127 # sequence length

dataset = GBMDataset(M, N, S0, r, sigma, T)
paths = dataset.data
paths = dataset.inverse_transform(paths).squeeze(1)

# Compute Monte Carlo option prices
call_price_mc = monte_carlo_option_price(paths, K, T, r,option_type="call")
put_price_mc = monte_carlo_option_price(paths, K, T, r,option_type="put")

# Compute Black-Scholes prices
call_price_bs = black_scholes_price(S0, K, T, r, sigma, option_type="call")
put_price_bs = black_scholes_price(S0, K, T, r, sigma, option_type="put")

# Print results
print(f"Monte Carlo Call Option Price: {call_price_mc:.4f}")
print(f"Black-Scholes Call Option Price: {call_price_bs:.4f}")
print(f"Monte Carlo Put Option Price: {put_price_mc:.4f}")
print(f"Black-Scholes Put Option Price: {put_price_bs:.4f}")

diffusion = train_diffusion_model(n_epochs=25, batch_size=64, device=device)
