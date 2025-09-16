import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


class TimeSeriesMetrics:
    @staticmethod
    def calculate_metrics(real_paths, generated_paths, dt):
        """Calculate statistical metrics and estimate GBM parameters (mu, sigma) for each path"""
        metrics = {}
        
        # Convert to numpy if needed
        if isinstance(real_paths, torch.Tensor):
            real_paths = real_paths.cpu().numpy()
        if isinstance(generated_paths, torch.Tensor):
            generated_paths = generated_paths.cpu().numpy()
        
        # Calculate log returns
        real_log_returns = np.log(real_paths[:, 1:] / real_paths[:, :-1])
        gen_log_returns = np.log(generated_paths[:, 1:] / generated_paths[:, :-1])
        
        # Estimate mu and sigma for each individual path
        def estimate_gbm_params(log_returns, dt):
            mu = np.mean(log_returns, axis=1) / dt + 0.5 * np.var(log_returns, axis=1, ddof=1) / dt
            sigma = np.sqrt(np.var(log_returns, axis=1, ddof=1) / dt)
            return mu, sigma
        
        real_mu, real_sigma = estimate_gbm_params(real_log_returns, dt)
        gen_mu, gen_sigma = estimate_gbm_params(gen_log_returns, dt)
        
        # Store individual estimates
        metrics['individual_gbm_params'] = {
            'real_mu': real_mu,
            'real_sigma': real_sigma,
            'generated_mu': gen_mu,
            'generated_sigma': gen_sigma
        }
        
        # Aggregate statistics (mean of individual estimates)
        metrics['gbm_params'] = {
            'real_mu': np.mean(real_mu),
            'real_sigma': np.mean(real_sigma),
            'generated_mu': np.mean(gen_mu),
            'generated_sigma': np.mean(gen_sigma)
        }
        
        # Log-return moments (for comparison)
        metrics['log_returns'] = {
            'real_mean': np.mean(real_log_returns),
            'generated_mean': np.mean(gen_log_returns),
            'real_std': np.std(real_log_returns),
            'generated_std': np.std(gen_log_returns),
            'real_skewness': stats.skew(real_log_returns.flatten()),
            'generated_skewness': stats.skew(gen_log_returns.flatten()),
            'real_kurtosis': stats.kurtosis(real_log_returns.flatten()),
            'generated_kurtosis': stats.kurtosis(gen_log_returns.flatten())
        }
        
        # Autocorrelation and KS test (unchanged)
        metrics['autocorr'] = {
            'real': np.mean([np.corrcoef(path[:-1], path[1:])[0,1] for path in real_log_returns]),
            'generated': np.mean([np.corrcoef(path[:-1], path[1:])[0,1] for path in gen_log_returns])
        }
        ks_stat, p_value = stats.ks_2samp(real_log_returns.flatten(), gen_log_returns.flatten())
        metrics['ks_test'] = {'statistic': ks_stat, 'p_value': p_value}
        
        return metrics

def plot_metrics_comparison(real_paths, generated_paths, title, dt):
    """Plot comparison including distributions of mu and sigma estimates"""
    metrics = TimeSeriesMetrics.calculate_metrics(real_paths, generated_paths, dt)
    
    fig = plt.figure(figsize=(24, 16))
    
    # Plot 1: Sample paths
    ax1 = plt.subplot(3, 2, 1)
    for i in range(min(5, len(real_paths))):
        plt.plot(real_paths[i], alpha=0.5, label=f'Real {i+1}')
    for i in range(min(5, len(generated_paths))):
        plt.plot(generated_paths[i], '--', alpha=0.5, label=f'Generated {i+1}')
    plt.title('Sample Paths')
    plt.legend()
    
    # Plot 2: Log returns distribution
    ax2 = plt.subplot(3, 2, 2)
    real_log_returns = np.log(real_paths[:, 1:] / real_paths[:, :-1])
    gen_log_returns = np.log(generated_paths[:, 1:] / generated_paths[:, :-1])
    plt.hist(real_log_returns.flatten(), bins=50, alpha=0.5, density=True, label='Real')
    plt.hist(gen_log_returns.flatten(), bins=50, alpha=0.5, density=True, label='Generated')
    plt.title('Log Returns Distribution')
    plt.legend()
    
    # Plot 3-4: Q-Q plots (unchanged)
    ax3 = plt.subplot(3, 2, 3)
    stats.probplot(real_log_returns.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Real)')
    
    ax4 = plt.subplot(3, 2, 4)
    stats.probplot(gen_log_returns.flatten(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Generated)')
    
    # Plot 5: Distribution of mu estimates
    ax5 = plt.subplot(3, 2, 5)
    plt.hist(metrics['individual_gbm_params']['real_mu'], bins=30, alpha=0.5, label='Real', density=True)
    plt.hist(metrics['individual_gbm_params']['generated_mu'], bins=30, alpha=0.5, label='Generated', density=True)
    plt.title('Distribution of μ Estimates')
    plt.legend()
    
    # Plot 6: Distribution of sigma estimates
    ax6 = plt.subplot(3, 2, 6)
    plt.hist(metrics['individual_gbm_params']['real_sigma'], bins=30, alpha=0.5, label='Real', density=True)
    plt.hist(metrics['individual_gbm_params']['generated_sigma'], bins=30, alpha=0.5, label='Generated', density=True)
    plt.title('Distribution of σ Estimates')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.close()
    
    # Print metrics (same as before)
    print("\nStatistical Metrics (Log Returns):")
    print(f"Mean: Real = {metrics['log_returns']['real_mean']:.6f}, Generated = {metrics['log_returns']['generated_mean']:.6f}")
    print(f"Std: Real = {metrics['log_returns']['real_std']:.6f}, Generated = {metrics['log_returns']['generated_std']:.6f}")
    
    print("\nGBM Parameter Estimates:")
    print(f"μ: Real = {metrics['gbm_params']['real_mu']:.6f}, Generated = {metrics['gbm_params']['generated_mu']:.6f}")
    print(f"σ: Real = {metrics['gbm_params']['real_sigma']:.6f}, Generated = {metrics['gbm_params']['generated_sigma']:.6f}")
    
    print("\nKS Test (Log Returns):")
    print(f"Statistic = {metrics['ks_test']['statistic']:.6f}, p-value = {metrics['ks_test']['p_value']:.6f}")
    
    return metrics

def monte_carlo_option_price(paths, K, T, r, n_timesteps, option_type="call"):
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


    # Compute option prices at each time step
    if option_type == "call":
        option_prices = S0 * stats.norm.cdf(d1) - K * np.exp(-r * time_to_maturity) * stats.norm.cdf(d2)
    elif option_type == "put":
        option_prices = K * np.exp(-r * time_to_maturity) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # At maturity, the option price is the payoff
    if option_type == "call":
        option_prices[0] = max(S0 - K, 0)
    elif option_type == "put":
        option_prices[0] = max(K - S0, 0)

    return option_prices


def plot_paths_and_prices(original_paths, diffusion_paths, S0, K, T, r, N, sigma, folder_path):
    # Compute call option prices at each time step
    call_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="call")
    call_prices_diffusion = monte_carlo_option_price(diffusion_paths, K, T, r, N, option_type="call")
    call_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="call")

    # Compute put option prices at each time step
    put_prices_mc = monte_carlo_option_price(original_paths, K, T, r, N, option_type="put")
    put_prices_diffusion = monte_carlo_option_price(diffusion_paths, K, T, r, N, option_type="put")
    put_prices_bs = black_scholes_price(S0, K, T, r, sigma, n_timesteps=N, option_type="put")

    # Plot: Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    t = np.linspace(0, T, N+1)  # Time array
    # Plot call option prices
    ax1.plot(t, call_prices_mc, label="Monte Carlo Call Price - actual", color="blue", linestyle="--")
    ax1.plot(t, call_prices_diffusion, label="Monte Carlo Call Price - diffusion", color="orange", linestyle="--")
    # ax1.plot(t, call_prices_bs, label="Black-Scholes Call Price", color="red", linestyle="-")
    ax1.set_title("European Call Option Price at Each Time Step")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Option Price")
    ax1.legend()
    ax1.grid(True)

    # Plot put option prices
    ax2.plot(t, put_prices_mc, label="Monte Carlo Put Price - actual", color="blue", linestyle="--")
    ax2.plot(t, put_prices_diffusion, label="Monte Carlo Put Price - diffusion", color="orange", linestyle="--")
    # ax2.plot(t, put_prices_bs, label="Black-Scholes Put Price", color="red", linestyle="-")
    ax2.set_title("European Put Option Price at Each Time Step")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Option Price")
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the combined plot
    plt.savefig(f'{folder_path}/combined_prices_mu={r}_sigma={sigma}_K={K}.png')
    # Close the figure to free up memory
    plt.close()

    # Compare the 2 methods: Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
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

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f'{folder_path}/generated_paths_mu={r}_sigma={sigma}.png')

    # Close the figure to free up memory
    plt.close()
