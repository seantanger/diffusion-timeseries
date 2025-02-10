import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


class TimeSeriesMetrics:
    @staticmethod
    def calculate_metrics(real_paths, generated_paths):
        """Calculate statistical metrics for comparison"""
        metrics = {}
        
        # Convert to numpy if needed
        if isinstance(real_paths, torch.Tensor):
            real_paths = real_paths.cpu().numpy()
        if isinstance(generated_paths, torch.Tensor):
            generated_paths = generated_paths.cpu().numpy()
        
        # Calculate returns (percentage changes)
        real_returns = np.diff(real_paths, axis=1) / real_paths[:, :-1]
        gen_returns = np.diff(generated_paths, axis=1) / generated_paths[:, :-1]
        
        # Basic statistical moments
        metrics['mean'] = {
            'real': np.mean(real_returns),
            'generated': np.mean(gen_returns)
        }
        metrics['std'] = {
            'real': np.std(real_returns),
            'generated': np.std(gen_returns)
        }
        metrics['skewness'] = {
            'real': stats.skew(real_returns.flatten()),
            'generated': stats.skew(gen_returns.flatten())
        }
        metrics['kurtosis'] = {
            'real': stats.kurtosis(real_returns.flatten()),
            'generated': stats.kurtosis(gen_returns.flatten())
        }
        
        # Autocorrelation (lag-1)
        metrics['autocorr'] = {
            'real': np.mean([np.corrcoef(path[:-1], path[1:])[0,1] for path in real_returns]),
            'generated': np.mean([np.corrcoef(path[:-1], path[1:])[0,1] for path in gen_returns])
        }
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(real_returns.flatten(), gen_returns.flatten())
        metrics['ks_test'] = {
            'statistic': ks_stat,
            'p_value': p_value
        }
        
        return metrics

def plot_metrics_comparison(real_paths, generated_paths, title):
    """Plot detailed comparison of real and generated paths"""
    real_paths_orig = real_paths
    generated_paths_orig = generated_paths

    # Calculate metrics
    metrics = TimeSeriesMetrics.calculate_metrics(real_paths_orig, generated_paths_orig)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot sample paths
    ax1 = plt.subplot(2, 2, 1)
    for i in range(min(5, len(real_paths_orig))):
        plt.plot(real_paths_orig[i], alpha=0.5, label=f'Real {i+1}')
    for i in range(min(5, len(generated_paths_orig))):
        plt.plot(generated_paths_orig[i], '--', alpha=0.5, label=f'Generated {i+1}')
    plt.title('Sample Paths Comparison')
    plt.legend()
    
    # Plot returns distribution
    ax2 = plt.subplot(2, 2, 2)
    real_returns = np.diff(real_paths_orig) / real_paths_orig[:, :-1]
    gen_returns = np.diff(generated_paths_orig) / generated_paths_orig[:, :-1]
    plt.hist(real_returns.flatten(), bins=50, alpha=0.5, density=True, label='Real Returns')
    plt.hist(gen_returns.flatten(), bins=50, alpha=0.5, density=True, label='Generated Returns')
    plt.title('Returns Distribution')
    plt.legend()
    
    # Plot Q-Q plot
    ax3 = plt.subplot(2, 2, 3)
    stats.probplot(real_returns.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Real Returns)')
    
    ax4 = plt.subplot(2, 2, 4)
    stats.probplot(gen_returns.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Generated Returns)')
    
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.close()
    
    # Print metrics
    print("\nStatistical Metrics Comparison:")
    print(f"Mean Returns: Real = {metrics['mean']['real']:.6f}, Generated = {metrics['mean']['generated']:.6f}")
    print(f"Std Returns: Real = {metrics['std']['real']:.6f}, Generated = {metrics['std']['generated']:.6f}")
    print(f"Skewness: Real = {metrics['skewness']['real']:.6f}, Generated = {metrics['skewness']['generated']:.6f}")
    print(f"Kurtosis: Real = {metrics['kurtosis']['real']:.6f}, Generated = {metrics['kurtosis']['generated']:.6f}")
    print(f"Autocorrelation: Real = {metrics['autocorr']['real']:.6f}, Generated = {metrics['autocorr']['generated']:.6f}")
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"Statistic = {metrics['ks_test']['statistic']:.6f}")
    print(f"P-value = {metrics['ks_test']['p_value']:.6f}")
    
    return metrics

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

    time_to_maturity = t  # Shape: (N+1,) should be t or T-t

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
    time_to_maturity = t  # Shape: (N+1,)

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

def plot_paths_and_prices(original_paths, diffusion_paths, spiking_diffusion_paths, S0, K, T, r, N, sigma):
    # Compute call option prices at each time step
    call_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="call")
    call_prices_diffusion = monte_carlo_option_price(diffusion_paths, K, T, r, N, option_type="call")
    call_prices_spiking_diffusion = monte_carlo_option_price(spiking_diffusion_paths, K, T, r, N, option_type="call")
    call_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="call")

    # Compute put option prices at each time step
    put_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="put")
    put_prices_diffusion = monte_carlo_option_price(diffusion_paths, K, T, r, N, option_type="put")
    put_prices_spiking_diffusion = monte_carlo_option_price(spiking_diffusion_paths, K, T, r, N, option_type="put")
    put_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="put")

    # Plot: Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    t = np.linspace(0, T, N+1)  # Time array
    # Plot call option prices
    ax1.plot(t, call_prices_mc, label="Monte Carlo Call Price", color="blue", linestyle="--")
    ax1.plot(t, call_prices_diffusion, label="Monte Carlo Call Price - diffusion", color="orange", linestyle="--")
    ax1.plot(t, call_prices_spiking_diffusion, label="Monte Carlo Call Price - spiking diffusion", color="green", linestyle="--")
    ax1.plot(t, call_prices_bs, label="Black-Scholes Call Price", color="red", linestyle="-")
    ax1.set_title("European Call Option Price at Each Time Step")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Option Price")
    ax1.legend()
    ax1.grid(True)

    # Plot put option prices
    ax2.plot(t, put_prices_mc, label="Monte Carlo Put Price", color="blue", linestyle="--")
    ax2.plot(t, put_prices_diffusion, label="Monte Carlo Put Price - diffusion", color="orange", linestyle="--")
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
    plt.savefig(f'combined_prices_mu={r}_sigma={sigma}_K={K}.png')
    # Close the figure to free up memory
    plt.close()

    # Compare the 3 methods: Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    # Calculate global y-axis limits
    y_min = min(
        np.min(original_paths),
        np.min(diffusion_paths))
    y_max = max(
        np.max(original_paths),
        np.max(diffusion_paths))
    # Plot original GBM paths
    ax1.plot(t, original_paths.T, lw=1)  # Transpose paths to (M, N)
    ax1.set_title("Original GBM Paths")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Price")
    ax1.set_ylim(y_min, y_max)  # Set y-axis limits
    ax1.grid(True)

    # Plot diffusion-generated paths
    ax2.plot(t, diffusion_paths.T, lw=1)  # Transpose paths to (M, N)
    ax2.set_title("Diffusion-Generated Paths")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Price")
    ax2.set_ylim(y_min, y_max)  # Set y-axis limits
    ax2.grid(True)

    # Plot spiking diffusion-generated paths
    ax3.plot(t, spiking_diffusion_paths.T, lw=1)  # Transpose paths to (M, N)
    ax3.set_title("Spiking Diffusion-Generated Paths")
    ax3.set_xlabel("Time (t)")
    ax3.set_ylabel("Price")
    ax3.set_ylim(y_min, y_max)  # Set y-axis limits
    ax3.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'generated_paths_mu={r}_sigma={sigma}_K={K}.png')

    # Close the figure to free up memory
    plt.close()

def run_multiple_mc(dataset, diffusion_model, spiking_diffusion_model, S0, K, T, r, N, sigma, n_simulations, device):
    # Compute call and put option prices for the original dataset
    original_paths = dataset.data
    original_paths = dataset.inverse_transform(original_paths)#.squeeze(1)

    call_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="call")
    put_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="put")

    # Compute Black-Scholes prices
    call_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="call")
    put_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="put")

    # Initialize lists to store results from multiple simulations
    call_prices_diffusion_all = []
    call_prices_spiking_diffusion_all = []
    put_prices_diffusion_all = []
    put_prices_spiking_diffusion_all = []

    # Run multiple simulations
    for _ in range(n_simulations):
        # Generate paths using diffusion models
        diffusion_paths = diffusion_model.sample(n_samples=10000, batch_size=100, device=device)
        diffusion_paths = dataset.inverse_transform(diffusion_paths)#.squeeze(1)

        spiking_diffusion_paths = spiking_diffusion_model.sample(n_samples=10000, batch_size=100, device=device)
        spiking_diffusion_paths = dataset.inverse_transform(spiking_diffusion_paths)#.squeeze(1)

        # Compute Monte Carlo prices for the current simulation
        call_prices_diffusion = monte_carlo_option_price(diffusion_paths, K, T, r, N, option_type="call")
        call_prices_spiking_diffusion = monte_carlo_option_price(spiking_diffusion_paths, K, T, r, N, option_type="call")
        put_prices_diffusion = monte_carlo_option_price(diffusion_paths, K, T, r, N, option_type="put")
        put_prices_spiking_diffusion = monte_carlo_option_price(spiking_diffusion_paths, K, T, r, N, option_type="put")

        # Append results to the lists
        call_prices_diffusion_all.append(call_prices_diffusion)
        call_prices_spiking_diffusion_all.append(call_prices_spiking_diffusion)
        put_prices_diffusion_all.append(put_prices_diffusion)
        put_prices_spiking_diffusion_all.append(put_prices_spiking_diffusion)

    # Plot: Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    t = np.linspace(0, T, N + 1)  # Time array

    # Plot call option prices
    ax1.plot(t, call_prices_mc, label="Monte Carlo Call Price (Original)", color="blue", linestyle="--", alpha=0.8)
    for i in range(n_simulations):
        ax1.plot(t, call_prices_diffusion_all[i], color="orange", linestyle="--", alpha=0.3, label="Diffusion" if i == 0 else None)
        ax1.plot(t, call_prices_spiking_diffusion_all[i], color="green", linestyle="--", alpha=0.3, label="Spiking Diffusion" if i == 0 else None)
    ax1.plot(t, call_prices_bs, label="Black-Scholes Call Price", color="red", linestyle="-", alpha=0.8)
    ax1.set_title("European Call Option Price at Each Time Step")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Option Price")
    ax1.legend()
    ax1.grid(True)

    # Plot put option prices
    ax2.plot(t, put_prices_mc, label="Monte Carlo Put Price (Original)", color="blue", linestyle="--", alpha=0.8)
    for i in range(n_simulations):
        ax2.plot(t, put_prices_diffusion_all[i], color="orange", linestyle="--", alpha=0.3, label="Diffusion" if i == 0 else None)
        ax2.plot(t, put_prices_spiking_diffusion_all[i], color="green", linestyle="--", alpha=0.3, label="Spiking Diffusion" if i == 0 else None)
    ax2.plot(t, put_prices_bs, label="Black-Scholes Put Price", color="red", linestyle="-", alpha=0.8)
    ax2.set_title("European Put Option Price at Each Time Step")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Option Price")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'multi_combined_prices_mu={r}_sigma={sigma}_K={K}.png')

    # Close the figure to free up memory
    plt.close()