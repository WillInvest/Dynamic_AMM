import os
import multiprocessing
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO
from parallel_simulate import parallel_simulate_with_constant_fee_rate, parallel_simulate_with_rl_amm
from typing import Dict, List
from datetime import datetime

def main(trader_dir, maker_dir, iterations=30, verbose=False):
    # Create a directory for storing results
    results_dir = "final_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the trained traders
    trader_paths = {
        mc: os.path.join(trader_dir, f'market_competition_level_{mc:.2f}', 'best_model.zip')
        for mc in np.arange(0.02, 0.22, 0.02)
    }

    # Define the fee rates
    max_fee_rate = 0.005
    min_fee_rate = 0.0005
    num_slices = 10
    fee_rates = np.linspace(min_fee_rate, max_fee_rate, num_slices)
    
    # Define sigma values
    sigma_values = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    
    # Initialize dictionary to store the results
    results = {}

    # Use multiprocessing to parallelize the simulations for constant fee rates
    # seed_range = range(iterations)
    results_constant_fee = parallel_simulate_with_constant_fee_rate(trader_paths, fee_rates, sigma_values, iterations)

    # Process results for constant fee rates
    for (fee_rate, sigma), result_list in results_constant_fee.items():
        if sigma not in results:
            results[sigma] = {
                'constant': {
                    'total_pnls': {},
                    'total_fees': {},
                    'total_vols': {},
                    'total_price_distance': {},
                    'total_transactions': {}
                },
                'rl': {
                    'total_pnls_rl': [],
                    'total_fees_rl': [],
                    'total_vols_rl': [],
                    'total_dynamic_fee': [],
                    'total_price_distance_rl': [],
                    'total_transactions_rl': []
                }
            }
        # Assigns references to the corresponding dictionaries in results[sigma]['constant'] for easier access
        total_pnls_constant = results[sigma]['constant']['total_pnls']
        total_fees_constant = results[sigma]['constant']['total_fees']
        total_vols_constant = results[sigma]['constant']['total_vols']
        total_price_distance_constant = results[sigma]['constant']['total_price_distance']
        total_transactions_constant = results[sigma]['constant']['total_transactions']

        if fee_rate not in total_pnls_constant:
            total_pnls_constant[fee_rate] = []
            total_fees_constant[fee_rate] = []
            total_vols_constant[fee_rate] = []
            total_price_distance_constant[fee_rate] = []
            total_transactions_constant[fee_rate] = []

        for result in result_list:
            total_pnl, total_fee, total_vol, price_distance, total_transaction = result
            total_pnls_constant[fee_rate].append(sum(total_pnl.values())) # sum over all traders pnl, since total_pnl is a list of pnls for traders
            total_fees_constant[fee_rate].append(sum(total_fee.values()))
            total_vols_constant[fee_rate].append(sum(total_vol.values()))
            total_price_distance_constant[fee_rate].append(price_distance)
            total_transactions_constant[fee_rate].append(sum(total_transaction.values()))
    
    # Use multiprocessing to parallelize the simulations for RL agent
    results_rl = parallel_simulate_with_rl_amm(trader_paths, maker_dir, sigma_values, iterations)

    # Process results for RL agent
    for (fee_rate, sigma), result_list in results_rl.items():
        total_pnls_rl = results[sigma]['rl']['total_pnls_rl']
        total_fees_rl = results[sigma]['rl']['total_fees_rl']
        total_vols_rl = results[sigma]['rl']['total_vols_rl']
        total_dynamic_fee = results[sigma]['rl']['total_dynamic_fee']
        total_price_distance_rl = results[sigma]['rl']['total_price_distance_rl']
        total_transactions_rl = results[sigma]['rl']['total_transactions_rl']

        for result in result_list:
            total_pnl, total_fee, total_vol, dynamic_fees, price_distance, total_transaction = result
            total_pnls_rl.append(sum(total_pnl.values()))
            total_fees_rl.append(sum(total_fee.values()))
            total_vols_rl.append(sum(total_vol.values()))
            total_dynamic_fee.extend(dynamic_fees)
            total_price_distance_rl.append(price_distance)
            total_transactions_rl.append(sum(total_transaction.values()))

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

        # For RL strategy
        for i in range(len(data['rl']['total_pnls_rl'])):
            flattened_results.append({
                'sigma': sigma,
                'fee_rate': 'rl',
                'pnl': data['rl']['total_pnls_rl'][i],
                'fee': data['rl']['total_fees_rl'][i],
                'volume': data['rl']['total_vols_rl'][i],
                'price_distance': data['rl']['total_price_distance_rl'][i],
                'dynamic_fee': data['rl']['total_dynamic_fee'][i],
                'total_transactions': data['rl']['total_transactions_rl'][i]
            })

    # Convert to DataFrame
    df = pd.DataFrame(flattened_results)

    # Save to CSV
    csv_file_path = os.path.join(results_dir, f'all_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    trader_dir = f'{os.path.expanduser("~")}/AMM-Python/models/trader_model'
    maker_dir = f'{os.path.expanduser("~")}/AMM-Python/models/maker_model/rl_maker_3000000_steps.zip'
    main(trader_dir, maker_dir, iterations=30)