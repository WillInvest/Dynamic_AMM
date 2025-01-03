import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice
import multiprocessing as mp
from functools import partial
import polars as pl
from datetime import datetime
import os

# Create results directory
root_dir = '/home/shiftpub/Dynamic_AMM/results'
timestamp = datetime.now().strftime('%Y%m%d')
analysis_dir = f'{root_dir}/fee_analysis_{timestamp}/sqrt data bootstrap'
os.makedirs(analysis_dir, exist_ok=True)

# Second cell - load data lazily
lazy_df = pl.scan_parquet('/home/shiftpub/Dynamic_AMM/results/simulation_results_20241206_125923/merged_results_final.parquet')
lazy_df = lazy_df.with_columns(pl.col('fee_rate').round(3))
# Check unique sigmas
sigmas = lazy_df.select('sigma').unique().collect()
print("Available sigmas:", sorted(sigmas['sigma'].to_list()))

def bootstrap_single_estimate(seed, data=None, percentile=0.05):
    """Single bootstrap estimate for parallel processing"""
    np.random.seed(seed)
    boot_sample = choice(data, size=int(np.sqrt(len(data))), replace=True)
    return np.percentile(boot_sample, percentile)

def parallel_bootstrap_percentile(data, n_bootstrap=1000, percentile=0.05, n_cores=None):
    """Parallel bootstrap percentile estimation"""
    if n_cores is None:
        n_cores = mp.cpu_count() // 2
        
    bootstrap_func = partial(bootstrap_single_estimate, data=data, percentile=percentile)
    seeds = range(n_bootstrap)
    
    chunk_size = max(1, n_bootstrap // n_cores)
    with mp.Pool(n_cores) as pool:
        bootstrap_estimates = np.array(pool.map(bootstrap_func, seeds, chunksize=chunk_size))
    
    # Return full statistics of bootstrap estimates
    return {
        'boot_mean': np.mean(bootstrap_estimates),
        'boot_std': np.std(bootstrap_estimates),
        'boot_ci_lower': np.percentile(bootstrap_estimates, 2.5),
        'boot_ci_upper': np.percentile(bootstrap_estimates, 97.5)
    }
    
def main(PCT, NBOOT):
    # Create a 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Selected sigmas from 0.1 to 0.9
    selected_sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Get number of CPU cores
    n_cores = mp.cpu_count() // 2
    print(f"Using {n_cores} CPU cores")
    print(f"Analyzing {PCT}th percentile")
    print(f"\nOptimal fee rates for each sigma (bootstrap for {PCT}th percentile):")
    print("-" * 40)

    # Store all results for CSV
    all_stats = []

    # Process one sigma at a time
    for idx, sigma in enumerate(selected_sigmas):
        print(f"Processing σ = {sigma}")
        
        sigma_data = (lazy_df
            .filter(pl.col('sigma') == sigma)
            .collect()
        )
        
        results = []
        unique_fee_rates = sorted(sigma_data['fee_rate'].unique())
        
        for fee_rate in unique_fee_rates:
            fee_data = sigma_data.filter(pl.col('fee_rate') == fee_rate)
            values = fee_data['total_fee_dollar_value'].to_numpy()
            
            # Direct percentile calculation
            direct_percentile = np.percentile(values, PCT)
            
            # Get bootstrap statistics
            bootstrap_stats = parallel_bootstrap_percentile(
                values, 
                n_bootstrap=NBOOT,
                percentile=PCT,
                n_cores=n_cores
            )
            
            # Store all statistics
            stats_dict = {
                'sigma': sigma,
                'fee_rate': fee_rate,
                f'direct_{PCT}th': direct_percentile,
                'sample_size': len(values),
                'sample_mean': np.mean(values),
                'sample_std': np.std(values),
                **bootstrap_stats
            }
            
            results.append(stats_dict)
            all_stats.append(stats_dict)
            
            del values
            
        # Plot results
        result_df = pl.DataFrame(results)
        
        # Find maximum point based on bootstrap mean
        max_idx = result_df['boot_mean'].arg_max()
        max_fee_rate = result_df['fee_rate'][max_idx]
        max_return = result_df['boot_mean'][max_idx]
        
        # Print optimal fee rate
        print(f"σ = {sigma:.1f}: optimal fee rate = {max_fee_rate:.3f}")
        
        # Create plots
        axes[idx].plot(
            result_df['fee_rate'],
            result_df[f'direct_{PCT}th'],
            'b-',
            label=f'Direct {PCT}th percentile',
            marker='o'
        )
        axes[idx].plot(
            result_df['fee_rate'],
            result_df['boot_mean'],
            'r--',
            label=f'Bootstrap {PCT}th mean',
            marker='s'
        )
        
        # Add confidence interval
        axes[idx].fill_between(
            result_df['fee_rate'],
            result_df['boot_ci_lower'],
            result_df['boot_ci_upper'],
            alpha=0.2,
            color='red',
            label='95% CI'
        )
        
        # Mark the maximum point
        axes[idx].plot(max_fee_rate, max_return, 'k*', markersize=10,
                      label=f'Max at {max_fee_rate:.3f}')
        
        axes[idx].set_title(f'σ = {sigma}')
        axes[idx].set_xlabel('Fee Rate')
        axes[idx].set_ylabel(f'{PCT}th Percentile Fee Value ($)')
        axes[idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        axes[idx].legend()
        
        del sigma_data
        del results
        del result_df

    # Save the plot with percentile in filename
    plt.tight_layout()
    plt.savefig(f'{analysis_dir}/p{PCT}_bootstrap_percentile_results.png')
    plt.close()

    # Save statistics to CSV with percentile in filename
    stats_df = pl.DataFrame(all_stats)
    stats_df = stats_df.sort(['sigma', 'fee_rate'])
    stats_df.write_csv(f'{analysis_dir}/p{PCT}_percentile_statistics.csv')


if __name__ == '__main__':
    # Set percentile value (can be changed as needed)
    NBOOT = 5000  # Number of bootstrap iterations

    for PCT in [15]:
        main(PCT, NBOOT)
        print(f"\nAnalysis for {PCT}th percentile completed\n")
