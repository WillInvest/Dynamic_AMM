import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from tqdm import tqdm
import pandas as pd
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO, TD3
import time
from datetime import datetime
from dummy_simulate import *
from amm_plot import *
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the simulation function
def run_simulation_with_constant_fee_rate(fee_rate, sigma, config):
    return simulate_with_constant_fee_rate(fee_rate, sigma=sigma, config=config)

def run_simulation_with_dynamic_fee_rate(config):
    return simulate_with_dynamic_fee_rate(config=config)

def parallel_constant(iterations, config):
    # Create a directory for storing results
    results_dir = f"{os.path.expanduser('~')}/AMM-Python/results/dummy_results"
    os.makedirs(results_dir, exist_ok=True)

    # Define the fee rates
    max_fee_rate = 0.0205
    min_fee_rate = 0.0005
    fee_rates = np.round(np.arange(min_fee_rate, max_fee_rate, min_fee_rate), 4)
    
    # Define sigma values
    # sigma_values = np.round(np.arange(0.05, 0.1, 0.01), 3)
    sigma_values = [None]
    results = {}

    # Start parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        future_to_sim = {}

        # Schedule tasks for each sigma and fee_rate combination
        for sigma in sigma_values:
            total_pnls_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
            total_fees_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
            total_vols_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
            total_transactions_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
            total_price_distance_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
            sigma_values_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}

            # Submit the tasks for parallel execution
            for fee_rate in fee_rates:
                for _ in range(iterations):
                    future = executor.submit(run_simulation_with_constant_fee_rate, fee_rate, sigma, config)
                    future_to_sim[future] = (sigma, fee_rate)

            # Process the results as they are completed
            for future in tqdm(as_completed(future_to_sim), total=len(future_to_sim), desc=f'Sigma {sigma} parallel execution'):
                sigma, fee_rate = future_to_sim[future]
                total_pnl, total_fee, total_vol, price_distance, total_transaction, estimated_mean_annualized_sigma = future.result()
                total_pnls_constant[fee_rate].append(total_pnl)
                total_fees_constant[fee_rate].append(total_fee)
                total_vols_constant[fee_rate].append(total_vol)
                total_price_distance_constant[fee_rate].append(price_distance)
                total_transactions_constant[fee_rate].append(total_transaction)
                sigma_values_constant[fee_rate].append(estimated_mean_annualized_sigma)

            # Store the results for this sigma value in the unified results dictionary
            results[sigma] = {
                'constant': {
                    'total_pnls': total_pnls_constant,
                    'total_fees': total_fees_constant,
                    'total_vols': total_vols_constant,
                    'total_price_distance': total_price_distance_constant,
                    'total_transactions': total_transactions_constant,
                    'estimated_sigma_values': sigma_values_constant
                }
            }

    # Flatten results and save to CSV
    flattened_results = []

    for sigma, data in results.items():
        for fee_rate, pnls in data['constant']['total_pnls'].items():
            for i in range(len(pnls)):
                flattened_results.append({
                    'sigma': sigma,
                    'fee_rate': fee_rate,
                    'pnl': pnls[i],
                    'fee': data['constant']['total_fees'][fee_rate][i],
                    'volume': data['constant']['total_vols'][fee_rate][i],
                    'estimated_sigma': data['constant']['estimated_sigma_values'][fee_rate][i],
                    # 'price_distance': data['constant']['total_price_distance'][fee_rate][i],
                    # 'dynamic_fee': fee_rate,
                    # 'total_transactions': data['constant']['total_transactions'][fee_rate][i]
                })

    # Convert to DataFrame
    df = pd.DataFrame(flattened_results)

    # Save to CSV and use the timestamp as part of the filename
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"static_simulation_results_{time_stamp}.csv"
    csv_file_path = os.path.join(results_dir, file_name)
    df.to_csv(csv_file_path, index=False)
    
def parallel_dynamic(iterations, config):
    # Create a directory for storing results
    results_dir = f"{os.path.expanduser('~')}/AMM-Python/results/dummy_results"
    os.makedirs(results_dir, exist_ok=True)
    # Initialize containers for results
    total_pnls = []
    total_fees = []
    total_vols = []
    total_price_distances = []
    total_transactions = []
    sigma_values = []

    # Start parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = []

        # Schedule tasks for each iteration
        for _ in range(iterations):
            future = executor.submit(run_simulation_with_dynamic_fee_rate, config)
            futures.append(future)

        # Process the results as they are completed
        for future in tqdm(as_completed(futures), total=len(futures), desc='Parallel execution'):
            total_pnl, total_fee, total_vol, price_distance, total_transaction, estimated_mean_annualized_sigma = future.result()

            # Append results to the respective lists
            total_pnls.append(total_pnl)
            total_fees.append(total_fee)
            total_vols.append(total_vol)
            total_price_distances.append(price_distance)
            total_transactions.append(total_transaction)
            sigma_values.append(estimated_mean_annualized_sigma)

    # Flatten results and prepare for DataFrame creation
    flattened_results = []

    for i in range(len(total_pnls)):
        flattened_results.append({
            'iteration': i,
            'pnl': total_pnls[i],
            'fee': total_fees[i],
            'volume': total_vols[i],
            'price_distance': total_price_distances[i],
            'transactions': total_transactions[i],
            'estimated_sigma': sigma_values[i]
        })

    # Convert to DataFrame for the main simulation results
    df = pd.DataFrame(flattened_results)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_file_name = f"dynamic_simulation_results_{time_stamp}.csv"
    main_csv_file_path = os.path.join(results_dir, main_file_name)
    df.to_csv(main_csv_file_path, index=False)


if __name__ == "__main__":
    
    '''
    S&P 500 Index
    - **Hours**: 6.5 hours
    - **Minutes**: \(6.5 \times 60 = 390\) minutes
    - **Seconds**: \(390 \times 60 = 23,400\) seconds
    Mean Daily Return: 0.0012640594294636917
    Std Daily Return: 0.007856416105203289
    '''
    
    import gc
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f'/home/shiftpub/AMM-Python/results/dynamic_fees/{time_stamp}'
    
    start_price=500
    mu=0.06
    dt=1/(252*6.5*60*60)
    steps=7800*60
    spread=0.005
        
    
    config = {
        'mu' : mu, 
        'spread' : spread,
        'dt' : dt,
        'start_price' : start_price,
        'steps' : steps,
        'save_folder' : save_folder
    }
    for iteration in [1000, 3000]:
        parallel_constant(iteration, config)
        gc.collect()
        parallel_dynamic(iteration, config)
        gc.collect()
