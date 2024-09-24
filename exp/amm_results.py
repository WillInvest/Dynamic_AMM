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
from amm_simulate import *
from amm_plot import *
from datetime import datetime

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
def main(trader_dir, maker_dir, iterations=300, verbose=False):

    # Create a directory for storing results
    results_dir = "final_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the trained traders
    traders = {}
    for mc in np.arange(0.05, 1.05, 0.05):
        model_path = os.path.join(trader_dir, f'market_competition_level_{mc:.2f}', 'best_model.zip')
        if os.path.exists(model_path):
            traders[mc] = TD3.load(model_path)
            print(f"Loaded model for market competition_level {mc:.2f}")

    # define the fee rates
    max_fee_rate = 0.005
    min_fee_rate = 0.0005
    num_slices = 10
    fee_rates = np.linspace(min_fee_rate, max_fee_rate, num_slices)
    
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
        for fee_rate in total_pnls_constant.keys():
            for seed in tqdm(range(iterations), desc=f'Sigma {sigma}, Fee Rate {fee_rate}'):
                seed = seed + int(time.time())
                total_pnl, total_fee, total_vol, price_distance, total_transaction, trader_transaction = simulate_with_constant_fee_rate(traders, fee_rate, seed, sigma=sigma)
                total_pnl_sum = sum(total_pnl.values())  # Sum the PnL of all traders
                total_fee_sum = sum(total_fee.values())  # Sum the fee of all traders
                total_vol_sum = sum(total_vol.values())  # Sum the volume of all traders
                total_transaction_sum = sum(total_transaction.values())  # Sum the volume of all traders
                total_pnls_constant[fee_rate].append(total_pnl_sum)
                total_fees_constant[fee_rate].append(total_fee_sum)
                total_vols_constant[fee_rate].append(total_vol_sum)
                total_price_distance_constant[fee_rate].append(price_distance)
                total_transactions_constant[fee_rate].append(total_transaction_sum)

                # Save intermediate trader_transaction to CSV
                trader_transaction_df = pd.DataFrame(trader_transaction)
                trader_transaction_csv_path = os.path.join(results_dir, f"intermediate_results_{time_stamp}", f"sigma_{sigma}", f"fee_{fee_rate}", 
                                                           f'trader_transactions_sigma_{sigma}_fee_{fee_rate}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                os.makedirs(os.path.dirname(trader_transaction_csv_path), exist_ok=True)
                trader_transaction_df.to_csv(trader_transaction_csv_path, index=False)
                
                # Save intermediate results for this fee rate and sigma
                intermediate_results = pd.DataFrame({
                    'sigma': [sigma],
                    'fee_rate': [fee_rate],
                    'pnl': [total_pnl_sum],
                    'fee': [total_fee_sum],
                    'volume': [total_vol_sum],
                    'price_distance': [price_distance],
                    'total_transactions': [total_transaction_sum]
                })
                csv_file_path = os.path.join(results_dir, f"intermediate_results_{time_stamp}", f"sigma_{sigma}", f"fee_{fee_rate}", 
                                             f'intermediate_results_sigma_{sigma}_fee_{fee_rate}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
                # Append to the file if it exists, otherwise create a new one
                if os.path.exists(csv_file_path):
                    intermediate_results.to_csv(csv_file_path, mode='a', header=False, index=False)
                else:
                    intermediate_results.to_csv(csv_file_path, index=False)

                if verbose:
                    print(f"Sigma {sigma}, Seed {seed}: Total PnL for fee rate {fee_rate}: {total_pnl_sum}")
                    print(f"Sigma {sigma}, Seed {seed}: Total Fee for fee rate {fee_rate}: {total_fee_sum}")
                    print(f"Sigma {sigma}, Seed {seed}: Total Volume for fee rate {fee_rate}: {total_vol_sum}")
                    print(f"Sigma {sigma}, Seed {seed}: Total price_distance for fee rate {fee_rate}: {price_distance}")
                    print(f"Sigma {sigma}, Seed {seed}: Total transactions for fee rate {fee_rate}: {total_transaction_sum}")
            
        # Initialize lists to collect results for RL agent
        total_pnls_rl = []
        total_fees_rl = []
        total_vols_rl = []
        total_dynamic_fee = []
        total_price_distance_rl = []
        total_transactions_rl = []

        # Collect results for RL agent
        for seed in tqdm(range(iterations), desc=f'Sigma {sigma}, RL Agent'):
            seed = seed + int(time.time())
            total_pnl, total_fee, total_vol, dynamic_fees, price_distance, total_transaction, trader_transaction = simulate_with_rl_amm(traders, seed, maker_dir, sigma=sigma)
            total_pnl_sum = sum(total_pnl.values())  # Sum the PnL of all traders
            total_fee_sum = sum(total_fee.values())  # Sum the fee of all traders
            total_vol_sum = sum(total_vol.values())  # Sum the volume of all traders
            total_transaction_sum = sum(total_transaction.values())  # Sum the volume of all traders
            total_pnls_rl.append(total_pnl_sum)
            total_fees_rl.append(total_fee_sum)
            total_vols_rl.append(total_vol_sum)
            total_dynamic_fee.extend(dynamic_fees)
            total_price_distance_rl.append(price_distance)
            total_transactions_rl.append(total_transaction_sum)
            
            # Save intermediate trader_transaction for RL agent to CSV
            trader_transaction_df_rl = pd.DataFrame(trader_transaction)
            trader_transaction_csv_path_rl = os.path.join(results_dir, f"intermediate_results_{time_stamp}", f"sigma_{sigma}", f"fee_rl", 
                                                          f'trader_transactions_sigma_{sigma}_fee_rl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            os.makedirs(os.path.dirname(trader_transaction_csv_path_rl), exist_ok=True)
            trader_transaction_df_rl.to_csv(trader_transaction_csv_path_rl, index=False)

            # Save intermediate results for RL agent
            intermediate_results_rl = pd.DataFrame({
                'sigma': [sigma],
                'fee_rate': ['rl'],
                'pnl': [total_pnl_sum],
                'fee': [total_fee_sum],
                'volume': [total_vol_sum],
                'price_distance': [price_distance],
                'dynamic_fee': [np.mean(dynamic_fees)],
                'total_transactions': [total_transaction_sum]
            })
            csv_file_path_rl = os.path.join(results_dir, f"intermediate_results_{time_stamp}", f"sigma_{sigma}", f"fee_rl", 
                                                      f'intermediate_results_sigma_{sigma}_fee_rl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            os.makedirs(os.path.dirname(csv_file_path_rl), exist_ok=True)
            
            # Append to the file if it exists, otherwise create a new one
            if os.path.exists(csv_file_path_rl):
                intermediate_results_rl.to_csv(csv_file_path_rl, mode='a', header=False, index=False)
            else:
                intermediate_results_rl.to_csv(csv_file_path_rl, index=False)

            if verbose:
                print(f"Sigma {sigma}, Seed {seed}: Total PnL for RL: {total_pnl_sum}")
                print(f"Sigma {sigma}, Seed {seed}: Total Fee for RL: {total_fee_sum}")
                print(f"Sigma {sigma}, Seed {seed}: Total Volume for RL: {total_vol_sum}")
                print(f"Sigma {sigma}, Seed {seed}: Dynamic Fee Rate: {np.mean(dynamic_fees)}")
                print(f"Sigma {sigma}, Seed {seed}: Total price_distance for RL: {price_distance}")
                print(f"Sigma {sigma}, Seed {seed}: Total transactions for RL: {total_transaction_sum}")

        # Store the results for this sigma value in the unified results dictionary
        results[sigma] = {
            'constant': {
                'total_pnls': total_pnls_constant,
                'total_fees': total_fees_constant,
                'total_vols': total_vols_constant,
                'total_price_distance': total_price_distance_constant,
                'total_transactions': total_transactions_constant
            },
            'rl': {
                'total_pnls_rl': total_pnls_rl,
                'total_fees_rl': total_fees_rl,
                'total_vols_rl': total_vols_rl,
                'total_dynamic_fee': total_dynamic_fee,
                'total_price_distance_rl': total_price_distance_rl,
                'total_transactions_rl': total_transactions_rl
            }
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

    # Save to CSV and use the timestamp as part of the filename
    csv_file_path = os.path.join(results_dir, f'all_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df.to_csv(csv_file_path, index=False)
    
if __name__ == "__main__":
    trader_dir = f'{os.path.expanduser("~")}/AMM-Python/models/trader_model'
    maker_dir = f'{os.path.expanduser("~")}/AMM-Python/models/maker_model/best_model.zip'
    main(trader_dir, maker_dir, iterations=30)