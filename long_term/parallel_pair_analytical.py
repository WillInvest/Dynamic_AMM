import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from theta_class import AMMStationaryDistributionFast
from scipy.stats import lognorm

# Global variable to store the model in each worker process
_worker_model = None
_worker_params = None

# --- Single step fee computation ---
def compute_single_step_fee(model, mu, sigma, dt, step, S0, n_grid=2000):
    """
    Compute the expected fee for a single step.
    
    Parameters:
    -----------
    model : AMMStationaryDistributionFast
        The AMM model
    mu, sigma, dt : float
        GBM parameters
    step : int
        Current step number
    S0 : float
        Initial price
    n_grid : int
        Number of grid points for integration
    
    Returns:
    --------
    float: Expected fee for this single step
    """
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
    return val


def init_worker(params):
    """
    Initialize worker process with shared model for a specific (sigma, gamma) pair.
    This is called once per worker process.
    """
    global _worker_model, _worker_params
    
    # Unpack params
    mu, sigma, gamma, dt = (
        params["mu"],
        params["sigma"],
        params["gamma"],
        params["dt"]
    )
    
    # Build stationary theta model once per worker
    bins = np.linspace(-1.0, 1.0, params["N_bins"])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    _worker_model = AMMStationaryDistributionFast(
        gamma=gamma, mu=mu, sigma=sigma, dt=dt, bins=bins, bin_centers=bin_centers
    )
    _worker_params = params


def process_sigma_gamma_pair(args):
    """
    Process a single (sigma, gamma) pair - compute fees for all steps with early stopping.
    
    Parameters:
    -----------
    args : tuple
        (sigma, gamma, total_steps, base_params)
    
    Returns:
    --------
    dict with results
    """
    sigma, gamma, total_steps, base_params = args
    
    try:
        # Create params for this specific pair
        params = base_params.copy()
        params["sigma"] = sigma
        params["gamma"] = gamma
        
        # Initialize model for this pair
        mu = params["mu"]
        dt = params["dt"]
        bins = np.linspace(-1.0, 1.0, params["N_bins"])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        model = AMMStationaryDistributionFast(
            gamma=gamma, mu=mu, sigma=sigma, dt=dt, bins=bins, bin_centers=bin_centers
        )
        
        # Store step fees
        accumulated_fee = 0.0
        previous_fee = None
        stopped_early = False
        stopped_step = total_steps
        
        # Compute fees for each step
        for step in range(1, total_steps + 1):
            # Compute single step fee
            current_fee = compute_single_step_fee(model, mu, sigma, dt, step, S0=1.0)
            
            # Early stopping: if current fee > previous fee, stop
            if previous_fee is not None and current_fee > previous_fee:
                stopped_early = True
                stopped_step = step
                break
            
            accumulated_fee += current_fee
            previous_fee = current_fee
        
        return {
            'sigma': sigma,
            'gamma': gamma,
            'total_steps_computed': stopped_step,
            'stopped_early': stopped_early,
            'accumulated_fee': accumulated_fee
        }
        
    except Exception as e:
        print(f"Error processing sigma={sigma}, gamma={gamma}: {e}")
        return None


def main(sigma_list, gamma_list, total_steps=1_000_000, 
         base_params=None, n_workers=None,
         output_file='results_pair_analytical.csv'):
    """
    Run parallel analytical computation over sigma and gamma pairs.
    
    Parameters:
    -----------
    sigma_list : list
        List of sigma values to test
    gamma_list : list
        List of gamma (fee rate) values to test
    total_steps : int
        Maximum total steps to compute (default: 1,000,000)
    base_params : dict
        Base parameters (mu, dt, N_bins). sigma and gamma will be overridden.
    n_workers : int, optional
        Number of parallel workers. If None, uses cpu_count()
    output_file : str
        Output CSV filename
    """
    if base_params is None:
        base_params = {
            "mu": 0.0,
            "dt": 12 / (365 * 24 * 60 * 60),
            "N_bins": 500,
        }
    
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Using {n_workers} parallel workers")
    print(f"Testing {len(sigma_list)} sigma values and {len(gamma_list)} gamma values")
    print(f"Total pairs: {len(sigma_list) * len(gamma_list)}")
    print(f"Total steps per pair: {total_steps}")
    
    # Prepare all tasks (all combinations of sigma and gamma)
    tasks = []
    for sigma in sigma_list:
        for gamma in gamma_list:
            tasks.append((sigma, gamma, total_steps, base_params))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Process in parallel
    all_results = []
    with Pool(n_workers) as pool:
        # Use imap for progress tracking
        for result in tqdm(
            pool.imap(process_sigma_gamma_pair, tasks),
            total=len(tasks),
            desc="Processing sigma-gamma pairs"
        ):
            if result is not None:
                all_results.append(result)
    
    # Flatten results: extract step_fees from each result
    summary_list = []
    
    for result in all_results:
        # Add summary info
        summary_list.append({
            'sigma': result['sigma'],
            'gamma': result['gamma'],
            'total_steps_computed': result['total_steps_computed'],
            'stopped_early': result['stopped_early'],
            'accumulated_fee': result['accumulated_fee']
        })
        
    
    # Create DataFrames
    summary_df = pd.DataFrame(summary_list)
    
    # Save results
    summary_df.to_csv(output_file.replace('.csv', '_summary.csv'), index=False)
    
    print(f"\nCompleted! Processed {len(all_results)} pairs")
    print(f"Summary saved to {output_file.replace('.csv', '_summary.csv')}")
    print(f"\nSummary:")
    print(summary_df.head(10))
    
    return summary_df


if __name__ == "__main__":
    # Define parameter ranges
    sigma_list = [0.5, 1.0, 2.0, 5.0, 10.0]
    gamma_list = [0.0001, 0.0003, 0.001, 0.003]
    
    base_params = {
        "mu": 0.0,
        "dt": 12 / (365 * 24 * 60 * 60),
        "N_bins": 500,
    }
    
    # Run parallel computation
    summary_df = main(
        sigma_list=sigma_list,
        gamma_list=gamma_list,
        total_steps=1_000,
        base_params=base_params,
        n_workers=20,
        output_file='results_pair_analytical.csv'
    )