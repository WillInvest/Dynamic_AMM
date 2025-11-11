import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from theta_class import AMMStationaryDistributionFast
from scipy.stats import lognorm

# Global variable to store the model in each worker process
_worker_model = None
_worker_params = None

# --- Analytical expectation via integration ---
def expected_total_fee_over_future_fast(model, mu, sigma, dt, start_step, total_step, S0, n_grid=2000, progress_bar=False):
    total_exp_fee = 0.0
    for step in range(start_step+1, total_step+1):
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


def init_worker(params):
    """
    Initialize worker process with shared model.
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


def compute_analytical_fees(total_steps, start_step, progress_bar=False):
    """
    Compute analytical fee expectations for total_steps.
    Uses the shared model from the worker process.
    
    Parameters:
    -----------
    total_steps : int
        Total number of steps
    start_step : int
        Starting step for fee accumulation
    progress_bar : bool
        Whether to show progress bar (not used in parallel execution)
    
    Returns:
    --------
    tuple: (analytical_exp_fee, static_multiple_step_fee, naive_multiple_step_fee)
    """
    global _worker_model, _worker_params
    
    if _worker_model is None:
        raise RuntimeError("Worker model not initialized. Call init_worker first.")
    
    mu = _worker_params["mu"]
    sigma = _worker_params["sigma"]
    dt = _worker_params["dt"]
    
    # --- Analytical expectation ---
    analytical_exp_fee = expected_total_fee_over_future_fast(
        _worker_model, mu, sigma, dt, start_step, total_steps, S0=1.0, progress_bar=progress_bar
    )
    static_multiple_step_fee = _worker_model._expected_incoming_fee_vec() * (total_steps - start_step)
    naive_multiple_step_fee = _worker_model._incoming_fee_vec(theta=0.0) * (total_steps - start_step)
    
    return analytical_exp_fee, static_multiple_step_fee, naive_multiple_step_fee


def process_single_step(total_steps):
    """
    Process a single total_steps value - designed for parallel execution.
    Uses the shared model from the worker process.
    
    Parameters:
    -----------
    total_steps : int
        Total number of steps to process
    
    Returns:
    --------
    dict with results
    """
    start_step = total_steps - 1
    
    try:
        ana, stat, naiv = compute_analytical_fees(
            total_steps, start_step, progress_bar=False
        )
        return {
            "total_steps": total_steps,
            "ana": ana,
            "stat": stat,
            "naiv": naiv
        }
    except Exception as e:
        print(f"Error processing total_steps={total_steps}: {e}")
        return None


def main(params, start=100, end=5_000_001, step=100, n_workers=None, output_file='results111.csv'):
    """
    Run parallel analytical computation.
    
    Parameters:
    -----------
    params : dict
        Model parameters (mu, sigma, gamma, dt, N_bins)
    start : int
        Starting total_steps value
    end : int
        Ending total_steps value (exclusive)
    step : int
        Step size for total_steps range
    n_workers : int, optional
        Number of parallel workers. If None, uses cpu_count()
    output_file : str
        Output CSV filename
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    print(f"Using {n_workers} parallel workers")
    print(f"Processing total_steps from {start} to {end} (step={step})")
    
    # Prepare all tasks (just the total_steps values, params will be passed via initializer)
    total_steps_list = list(np.arange(start, end, step))
    
    print(f"Total tasks: {len(total_steps_list)}")
    
    # Process in parallel with worker initialization
    results = []
    with Pool(n_workers, initializer=init_worker, initargs=(params,)) as pool:
        # Use imap for progress tracking
        for result in tqdm(
            pool.imap(process_single_step, total_steps_list),
            total=len(total_steps_list),
            desc="Processing analytical computations"
        ):
            if result is not None:
                results.append(result)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_steps')  # Ensure sorted by total_steps
    results_df.to_csv(output_file, index=False)
    
    print(f"\nCompleted! Processed {len(results_df)} data points")
    print(f"Results saved to {output_file}")
    print(f"\nFirst few rows:")
    print(results_df.head(10))
    
    return results_df


if __name__ == "__main__":
    params = {
        "mu": 0.0,
        "sigma": 1,
        "gamma": 0.0003,
        "dt": 12 / (365 * 24 * 60 * 60),
        "N_bins": 500,
    }

    start_step = 100
    end_step = 1_000_001
    # Run parallel computation
    # Adjust n_workers if needed (defaults to all available cores)
    main(params, start=start_step, end=end_step, step=1000, n_workers=None, output_file=f'results_sigma_{params["sigma"]}_{start_step}_{end_step}.csv')

