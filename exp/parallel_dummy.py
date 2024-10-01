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
def run_simulation(fee_rate, sigma, seed, config):
    seed = seed + int(time.time())
    return simulate_with_constant_fee_rate(fee_rate, seed, sigma=sigma, config=config)

def main(maker_dir, iterations, config):
    # Create a directory for storing results
    results_dir = f"{os.path.expanduser('~')}/AMM-Python/results/dummy_results"
    os.makedirs(results_dir, exist_ok=True)

    # Define the fee rates
    max_fee_rate = 0.02
    min_fee_rate = 0.0005
    fee_rates = np.round(np.arange(min_fee_rate, max_fee_rate, min_fee_rate), 4)
    
    # Define sigma values
    sigma_values = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
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

            # Submit the tasks for parallel execution
            for fee_rate in fee_rates:
                for seed in range(iterations):
                    future = executor.submit(run_simulation, fee_rate, sigma, seed, config)
                    future_to_sim[future] = (sigma, fee_rate)

            # Process the results as they are completed
            for future in tqdm(as_completed(future_to_sim), total=len(future_to_sim), desc=f'Sigma {sigma} parallel execution'):
                sigma, fee_rate = future_to_sim[future]
                total_pnl, total_fee, total_vol, price_distance, total_transaction = future.result()
                total_pnls_constant[fee_rate].append(total_pnl)
                total_fees_constant[fee_rate].append(total_fee)
                total_vols_constant[fee_rate].append(total_vol)
                total_price_distance_constant[fee_rate].append(price_distance)
                total_transactions_constant[fee_rate].append(total_transaction)

            # Store the results for this sigma value in the unified results dictionary
            results[sigma] = {
                'constant': {
                    'total_pnls': total_pnls_constant,
                    'total_fees': total_fees_constant,
                    'total_vols': total_vols_constant,
                    'total_price_distance': total_price_distance_constant,
                    'total_transactions': total_transactions_constant
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
                    'price_distance': data['constant']['total_price_distance'][fee_rate][i],
                    'dynamic_fee': fee_rate,
                    'total_transactions': data['constant']['total_transactions'][fee_rate][i]
                })

    # Convert to DataFrame
    df = pd.DataFrame(flattened_results)

    # Save to CSV and use the timestamp as part of the filename
    file_name = f"newa_results_spread_{config['spread']}_dt_{config['dt']:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file_path = os.path.join(results_dir, file_name)
    df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    dt = 1/3600
    mu = 0.0001
    start_price = 50000
    steps = 50000
    spread = 0.01
    
    config = {
        'mu' : mu, 
        'spread' : spread,
        'dt' : dt,
        'start_price' : start_price,
        'steps' : steps
    }
    maker_dir = f'{os.path.expanduser("~")}/AMM-Python/models/dummy_maker_model/rl_maker_17312000_steps.zip'
    main(maker_dir, iterations=50, config=config)