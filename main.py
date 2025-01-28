	
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
    
def monte_carlo_option_price(paths, K, T, r, n_timesteps, option_type="call"):
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
    t = np.linspace(0, T, n_timesteps+1)  # Time array
    # Time to maturity at each time step

    time_to_maturity = T - t  # Shape: (N+1,)

    # Compute payoffs at each time step
    if option_type == "call":
        payoffs = np.maximum(paths - K, 0)  # Shape: (M, N+1)
    elif option_type == "put":
        payoffs = np.maximum(K - paths, 0)  # Shape: (M, N+1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount payoffs to present value
    discount_factors = np.exp(-r * time_to_maturity)  # Shape: (N+1,)
    discounted_payoffs = payoffs * discount_factors  # Shape: (M, N+1)

    # Compute option price as the average of discounted payoffs
    option_prices = np.mean(discounted_payoffs, axis=0)  # Shape: (N+1,)

    return option_prices

def black_scholes_price(S0, K, T, r, sigma, n_timesteps, option_type="call"):
    """
    Compute the Black-Scholes price of a European option at each time step.

    Parameters:
    S0         : Initial price
    K          : Strike price
    T          : Time to maturity
    r          : Risk-free rate
    sigma      : Volatility
    option_type: "call" or "put"

    Returns:
    t          : Time array (N+1 array)
    option_prices : Black-Scholes option prices at each time step (N+1 array)
    """
    t = np.linspace(0, T, n_timesteps+1)  # Time array

    # Time to maturity at each time step
    time_to_maturity = T - t  # Shape: (N+1,)

    # Compute d1 and d2 for all time steps
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
        d2 = d1 - sigma * np.sqrt(time_to_maturity)

    # Handle the case at maturity (t = T)
    d1[-1] = np.inf if S0 > K else -np.inf
    d2[-1] = np.inf if S0 > K else -np.inf

    # Compute option prices at each time step
    if option_type == "call":
        option_prices = S0 * stats.norm.cdf(d1) - K * np.exp(-r * time_to_maturity) * stats.norm.cdf(d2)
    elif option_type == "put":
        option_prices = K * np.exp(-r * time_to_maturity) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # At maturity, the option price is the payoff
    if option_type == "call":
        option_prices[-1] = max(S0 - K, 0)
    elif option_type == "put":
        option_prices[-1] = max(K - S0, 0)

    return option_prices

def train_diffusion_model(dataset, n_epochs, batch_size, device, spiking):
    # Create dataset
    sequence_length = 127
    # dataset = GBMDataset(10000, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    if spiking:
        diffusion = SpikingDiffusionModel(sequence_length=sequence_length+1, device=device)
    else:
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
t = np.linspace(0, T, N+1)  # Time array
batch_size=32
sbatch_size=16

dataset = GBMDataset(n_samples=M, sequence_length=N, S0=S0, mu=r, sigma=sigma, T=T)
paths = dataset.data
paths = dataset.inverse_transform(paths).squeeze(1)


# Train diffusion models
diffusion, losses = train_diffusion_model(dataset=dataset, n_epochs=30, batch_size=batch_size, device=device, spiking=False)
torch.save(diffusion.model.state_dict(), f"./parameters/timeseries_unet_mu={r}_sigma={sigma}_t={T}")
spiking_diffusion, spiking_losses = train_diffusion_model(dataset=dataset, n_epochs=30, batch_size=sbatch_size, device=device, spiking=True)
torch.save(spiking_diffusion.model.state_dict(), f"./parameters/spiking_timeseries_unet_mu={r}_sigma={sigma}_t={T}")

# Generate paths
generated_paths = diffusion.sample(n_samples = M, batch_size= 32, device=device)
generated_paths_transformed = dataset.inverse_transform(generated_paths).squeeze(1)
spiking_generated_paths = spiking_diffusion.sample(n_samples = M, batch_size= 32, device=device)
spiking_generated_paths_transformed = dataset.inverse_transform(spiking_generated_paths).squeeze(1)

# Compute call option prices at each time step
call_prices_mc = monte_carlo_option_price(paths, K, T, r, N, option_type="call")
call_prices_diffusion = monte_carlo_option_price(generated_paths_transformed, K, T, r, N, option_type="call")
call_prices_spiking_diffusion = monte_carlo_option_price(spiking_generated_paths_transformed, K, T, r, N, option_type="call")
call_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="call")

# Compute put option prices at each time step
put_prices_mc = monte_carlo_option_price(paths, K, T, r, N, option_type="put")
put_prices_diffusion = monte_carlo_option_price(generated_paths_transformed, K, T, r, N, option_type="put")
put_prices_spiking_diffusion = monte_carlo_option_price(spiking_generated_paths_transformed, K, T, r, N, option_type="put")
put_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="put")

# Plot: Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot call option prices
ax1.plot(t, call_prices_mc, label="Monte Carlo Call Price", color="blue", linestyle="--")
ax1.plot(t, call_prices_diffusion, label="Monte Carlo Call Price - diffusion", color="yellow", linestyle="--")
ax1.plot(t, call_prices_spiking_diffusion, label="Monte Carlo Call Price - spiking diffusion", color="green", linestyle="--")
ax1.plot(t, call_prices_bs, label="Black-Scholes Call Price", color="red", linestyle="-")
ax1.set_title("European Call Option Price at Each Time Step")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Option Price")
ax1.legend()
ax1.grid(True)

# Plot put option prices
ax2.plot(t, put_prices_mc, label="Monte Carlo Put Price", color="blue", linestyle="--")
ax2.plot(t, put_prices_diffusion, label="Monte Carlo Put Price - diffusion", color="yellow", linestyle="--")
ax2.plot(t, put_prices_spiking_diffusion, label="Monte Carlo Put Price - spiking diffusion", color="green", linestyle="--")
ax2.plot(t, put_prices_bs, label="Black-Scholes Put Price", color="red", linestyle="-")
ax2.set_title("European Put Option Price at Each Time Step")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Option Price")
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
# plt.show()
# Save the combined plot
plt.savefig(f'combined_prices_mu={r}_sigma={sigma}_K={K}.pdf')
# Close the figure to free up memory
plt.close()


# Compare the 3 methods: Create a figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

# Plot original GBM paths
ax1.plot(t, paths.T, lw=1)  # Transpose paths to (M, N)
ax1.set_title("Original GBM Paths")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Price")
ax1.grid(True)

# Plot diffusion-generated paths
ax2.plot(t, generated_paths_transformed.T, lw=1)  # Transpose paths to (M, N)
ax2.set_title("Diffusion-Generated Paths")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Price")
ax2.grid(True)

# Plot spiking diffusion-generated paths
ax3.plot(t, spiking_generated_paths_transformed.T, lw=1)  # Transpose paths to (M, N)
ax3.set_title("Spiking Diffusion-Generated Paths")
ax3.set_xlabel("Time (t)")
ax3.set_ylabel("Price")
ax3.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined plot
plt.savefig(f'generated_paths_mu={r}_sigma={sigma}_K={K}.pdf')

# Close the figure to free up memory
plt.close()