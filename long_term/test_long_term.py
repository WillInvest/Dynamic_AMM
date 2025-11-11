
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad
from theta_class import AMMStationaryDistributionFast
from tqdm import tqdm
from numpy.random import PCG64, Generator

# --- Analytical expectation via integration ---
def expected_total_fee_over_future_fast(model, mu, sigma, dt, start_step, total_step, S0, n_grid=2000, progress_bar=True):
    total_exp_fee = 0.0
    for step in tqdm(range(start_step+1, total_step+1), disable=not progress_bar):
        if step == 0:
            continue
        t = step * dt
        mean_ln = np.log(S0) + (mu - 0.5 * sigma**2) * t
        sd_ln = sigma * np.sqrt(t)

        # Construct the s-grid (log-space, wide enough to cover tails)
        low = np.exp(mean_ln - 5 * sd_ln)
        high = np.exp(mean_ln + 5 * sd_ln)
        s_grid = np.linspace(low, high, n_grid)

        # Evaluate fee and PDF
        fee_vals = model._expected_incoming_fee_v3(s0=s_grid)
        pdf_vals = lognorm.pdf(s_grid, s=sd_ln, scale=np.exp(mean_ln))

        # Integrate numerically
        val = np.trapezoid(fee_vals * pdf_vals, s_grid)
        total_exp_fee += val
    return total_exp_fee

# --- AMM Simulation Function with Step-wise Recording ---
def run_amm_simulation_with_recording(S0, n_paths, total_steps, x0, y0, mu, sigma, gamma, dt, 
                                     record_interval=100, rng=None, show_progress=False):
    """
    Run AMM simulation and record mean/variance at regular intervals.
    
    Parameters:
    -----------
    record_interval : int
        Record statistics every N steps (e.g., 100 means record at steps 100, 200, 300, ...)
    show_progress : bool
        If True, show progress bar
    
    Returns:
    --------
    results : list of dict
        Each dict contains: {'steps': int, 'mean': float, 'variance': float}
    """
    # --- Sample theta and compute initial S0 per path ---
    s_current = S0.copy()
    if rng is None:
        rng = Generator(PCG64(123))
    L = np.sqrt(x0 * y0)
    
    # --- Initialize inventories and fee ---
    xt = np.full(n_paths, x0, dtype=float)
    yt = np.full(n_paths, y0, dtype=float)
    step_fee = np.zeros(n_paths)  # Fee collected at current step (reset each step)

    # --- Pre-compute constants for performance ---
    drift_factor = (mu - 0.5 * sigma**2) * dt
    volatility_factor = sigma * np.sqrt(dt)
    one_minus_gamma = 1.0 - gamma
    gamma_over_one_minus_gamma = gamma / one_minus_gamma
    sqrt_one_minus_gamma = np.sqrt(one_minus_gamma)

    # --- Determine recording steps ---
    recording_steps = set(range(record_interval, total_steps + 1, record_interval))
    results = []

    # --- Run simulation ---    
    iterator = tqdm(range(1, total_steps+1), desc=f"Simulation {total_steps} steps") if show_progress else range(1, total_steps+1)
    
    for i in iterator:
        # Reset step fee for this iteration
        step_fee.fill(0.0)
        
        # Generate random increments and update price incrementally
        z_pool = rng.normal(size=n_paths)
        s_current *= np.exp(drift_factor + volatility_factor * z_pool)
        
        P_AMM = yt / xt
        bid = P_AMM * one_minus_gamma
        ask = P_AMM / one_minus_gamma

        up_mask = s_current > ask 
        down_mask = s_current < bid

        if np.any(up_mask):
            s_up = s_current[up_mask]
            inv_sqrt = 1.0 / np.sqrt(one_minus_gamma * s_up)
            x_new = L * inv_sqrt
            y_new = L * sqrt_one_minus_gamma * np.sqrt(s_up)
            fee_up = gamma_over_one_minus_gamma * (y_new - yt[up_mask])
            xt[up_mask] = x_new
            yt[up_mask] = y_new
            step_fee[up_mask] += fee_up

        if np.any(down_mask):
            s_down = s_current[down_mask]
            sqrt_ratio = np.sqrt(one_minus_gamma / s_down)
            x_new = L * sqrt_ratio
            y_new = L * np.sqrt(s_down / one_minus_gamma)
            fee_down = gamma_over_one_minus_gamma * (x_new - xt[down_mask]) * s_down
            xt[down_mask] = x_new
            yt[down_mask] = y_new
            step_fee[down_mask] += fee_down

        # Record statistics at specified intervals
        if i in recording_steps:
            mean_fee = np.mean(step_fee)
            variance_fee = np.var(step_fee)
            results.append({
                'steps': i,
                'mean': mean_fee,
                'variance': variance_fee
            })
    
    return results


def main(params, total_steps=1_000_000, record_interval=100):
    """
    Run a single long simulation and record mean/variance at regular intervals.
    
    Parameters:
    -----------
    params : dict
        Simulation parameters
    total_steps : int
        Total number of simulation steps (default: 1,000,000)
    record_interval : int
        Record statistics every N steps (default: 100)
    """
    x0, y0, mu, sigma, gamma, dt, n_paths = (
        params["x0"], params["y0"], params["mu"],
        params["sigma"], params["gamma"], params["dt"],
        params["n_paths"]
    )

    # --- Build stationary theta model ---
    bins = np.linspace(-1.0, 1.0, params["N_bins"])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    model = AMMStationaryDistributionFast(
        gamma=gamma, mu=mu, sigma=sigma, dt=dt, bins=bins, bin_centers=bin_centers
    )
    
    # --- Generate initial S0 for all paths ---
    rng = Generator(PCG64(123))
    theta_list = np.concatenate([model.bin_centers, [-1.0, 1.0]])
    pi = model.pi_star / model.pi_star.sum()
    theta_samples = rng.choice(theta_list, size=n_paths, p=pi)
    S0 = (y0 / x0) * np.power(1.0 - gamma, -theta_samples)
    
    print(f"Initial S0 generated successfully for {n_paths:,} paths")
    print(f"Starting simulation with {total_steps:,} total steps, recording every {record_interval} steps...")
    
    # --- Run single simulation with recording ---
    results = run_amm_simulation_with_recording(
        S0, n_paths, total_steps, x0, y0, mu, sigma, gamma, dt,
        record_interval=record_interval, rng=rng, show_progress=True
    )
    
    # --- Convert to DataFrame and save ---
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('steps')  # Ensure sorted by steps
    results_df.to_csv(f'results_sigma_{sigma}_total_{total_steps}_record_{record_interval}.csv', index=False)
    
    print(f"\nSimulation completed! Recorded {len(results_df)} data points")
    print(f"Results saved to results_sigma_{sigma}_total_{total_steps}_record_{record_interval}.csv")
    print(f"\nFirst few rows:")
    print(results_df.head(10))
    
    return results_df

  

if __name__ == "__main__":
    import pandas as pd

    params = {
        "x0": 1e6,
        "y0": 1e6,
        "mu": 0.0,
        "sigma": 1,
        "gamma": 0.0003,
        "dt": 12 / (365 * 24 * 60 * 60),
        "N_bins": 500,
        "n_paths": 1_000_000,
    }

    # Run simulation: 1 million steps, record every 100 steps
    main(params, total_steps=1_000_000, record_interval=1000)