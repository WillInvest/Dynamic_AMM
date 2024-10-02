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

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
def main(maker_dir, iterations, config):

    # Create a directory for storing results
    results_dir = f"{os.path.expanduser('~')}/AMM-Python/results/dummy_results"
    os.makedirs(results_dir, exist_ok=True)

    # define the fee rates
    max_fee_rate = 0.2
    min_fee_rate = 0.0005
    fee_rates = np.round(np.arange(min_fee_rate, max_fee_rate, min_fee_rate), 4)
    
    # Define sigma values
    sigma_values = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    results = {}

    for sigma in sigma_values:
        # Initialize dictionaries to collect results for each sigma
        total_pnls_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
        total_fees_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
        total_vols_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
        total_transactions_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}
        total_price_distance_constant = {round(fee_rate, 4): [] for fee_rate in fee_rates}

        # Collect results for constant fee rates
        for fee_rate in fee_rates:
            fee_rate = np.round(fee_rate, 4)
            for seed in tqdm(range(iterations), desc=f'Sigma {sigma}, Fee Rate {fee_rate}'):
                seed = seed + int(time.time())
                total_pnl, total_fee, total_vol, price_distance, total_transaction = simulate_with_constant_fee_rate(fee_rate, seed, sigma=sigma, config=config)
                total_pnls_constant[fee_rate].append(total_pnl)
                total_fees_constant[fee_rate].append(total_fee)
                total_vols_constant[fee_rate].append(total_vol)
                total_price_distance_constant[fee_rate].append(price_distance)
                total_transactions_constant[fee_rate].append(total_transaction)
            
        # # Initialize lists to collect results for RL agent
        # total_pnls_rl = []
        # total_fees_rl = []
        # total_vols_rl = []
        # total_dynamic_fee = []
        # total_price_distance_rl = []
        # total_transactions_rl = []

        # # Collect results for RL agent
        # for seed in tqdm(range(iterations), desc=f'Sigma {sigma}, RL Agent'):
        #     seed = seed + int(time.time())
        #     total_pnl, total_fee, total_vol, dynamic_fees, price_distance, total_transaction = simulate_with_rl_amm(maker_dir, seed, sigma=sigma)
        #     total_pnls_rl.append(total_pnl)
        #     total_fees_rl.append(total_fee)
        #     total_vols_rl.append(total_vol)
        #     total_dynamic_fee.extend(dynamic_fees)
        #     total_price_distance_rl.append(price_distance)
        #     total_transactions_rl.append(total_transaction)

        # Store the results for this sigma value in the unified results dictionary
        results[sigma] = {
            'constant': {
                'total_pnls': total_pnls_constant,
                'total_fees': total_fees_constant,
                'total_vols': total_vols_constant,
                'total_price_distance': total_price_distance_constant,
                'total_transactions': total_transactions_constant
            }
            # 'rl': {
            #     'total_pnls_rl': total_pnls_rl,
            #     'total_fees_rl': total_fees_rl,
            #     'total_vols_rl': total_vols_rl,
            #     'total_dynamic_fee': total_dynamic_fee,
            #     'total_price_distance_rl': total_price_distance_rl,
            #     'total_transactions_rl': total_transactions_rl
            # }
        }

    # Flatten results and save to CSV
    flattened_results = []

    for sigma, data in results.items():
        # For constant fee rates
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

        # # For RL strategy
        # for i in range(len(data['rl']['total_pnls_rl'])):
        #     flattened_results.append({
        #         'sigma': sigma,
        #         'fee_rate': 'rl',
        #         'pnl': data['rl']['total_pnls_rl'][i],
        #         'fee': data['rl']['total_fees_rl'][i],
        #         'volume': data['rl']['total_vols_rl'][i],
        #         'price_distance': data['rl']['total_price_distance_rl'][i],
        #         'dynamic_fee': data['rl']['total_dynamic_fee'][i],
        #         'total_transactions': data['rl']['total_transactions_rl'][i]
        #     })

    # Convert to DataFrame
    df = pd.DataFrame(flattened_results)

    # Save to CSV and use the timestamp as part of the filename
    # make file name more descriptive containing the config parameters
    file_name = f"newa_results_spread_{config['spread']}_dt_{config['dt']:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file_path = os.path.join(results_dir, file_name)
    df.to_csv(csv_file_path, index=False)
    
if __name__ == "__main__":
    
    
    '''
    S&P 500 Index
    - **Hours**: 6.5 hours
    - **Minutes**: \(6.5 \times 60 = 390\) minutes
    - **Seconds**: \(390 \times 60 = 23,400\) seconds
    Mean Daily Return: 0.0012640594294636917
    Std Daily Return: 0.007856416105203289
    '''
    
    dt = 1/23400
    mu = 0.001
    start_price = 500
    steps = 5000
    spread = 0.5
    
    config = {
        'mu' : mu, 
        'spread' : spread,
        'dt' : dt,
        'start_price' : start_price,
        'steps' : steps
    }
    maker_dir = f'{os.path.expanduser("~")}/AMM-Python/models/dummy_maker_model/rl_maker_17312000_steps.zip'
    main(maker_dir, iterations=50, config=config)