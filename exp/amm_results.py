import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
import csv
import pandas as pd
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO
import time
from amm_simulate import *
from amm_plot import *

def save_merged_results_to_csv(results_dir: str, filename: str, constant_data: Dict[float, List[float]], rl_data: List[float]):
    """Save merged constant fee and RL data to a single CSV file."""
    filepath = os.path.join(results_dir, filename)
    max_length = max(max(len(v) for v in constant_data.values()), len(rl_data))
    
    with open(filepath, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        headers = [f'Fee Rate {fee_rate:.4f}' for fee_rate in constant_data.keys()] + ['RL-based AMM']
        writer.writerow(['Seed'] + headers)
        
        # Write data rows
        for i in range(max_length):
            row = [i]
            for fee_rate in constant_data.keys():
                row.append(constant_data[fee_rate][i] if i < len(constant_data[fee_rate]) else '')
            row.append(rl_data[i] if i < len(rl_data) else '')
            writer.writerow(row)
    
    print(f"Saved merged results to {filepath}")
    
    
def main(trader_dir, maker_dir, iterations=300):

    # Create a directory for storing results
    results_dir = "final_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the trained traders
    traders = {}
    for mc in np.arange(0.02, 0.22, 0.02):
        model_path = os.path.join(trader_dir, f'market_competition_level_{mc:.2f}', 'rl_model_1000000_steps.zip')
        if os.path.exists(model_path):
            traders[mc] = PPO.load(model_path)
            print(f"Loaded model for market competition_level {mc:.2f}")

    # define the fee rates
    max_fee_rate = 0.005
    min_fee_rate = 0.0005
    num_slices = 10
    fee_rates = np.linspace(min_fee_rate, max_fee_rate, num_slices)
    
    # Collect total PnL for constant fee rates
    total_pnls_constant = {round(fee_rate,4): [] for fee_rate in fee_rates}
    total_fees_constant = {round(fee_rate,4): [] for fee_rate in fee_rates}
    total_vols_constant = {round(fee_rate,4): [] for fee_rate in fee_rates}
    total_price_distance_constant = {round(fee_rate,4): [] for fee_rate in fee_rates}
    for fee_rate in total_pnls_constant.keys():
        for seed in range(iterations):
            seed = seed + int(time.time())
            total_pnl, total_fee, total_vol, price_distance = simulate_with_constant_fee_rate(traders, fee_rate, seed)
            total_pnl_sum = sum(total_pnl.values())  # Sum the PnL of all traders
            total_fee_sum = sum(total_fee.values())  # Sum the fee of all traders
            total_vol_sum = sum(total_vol.values())  # Sum the volume of all traders
            total_pnls_constant[fee_rate].append(total_pnl_sum)
            total_fees_constant[fee_rate].append(total_fee_sum)
            total_vols_constant[fee_rate].append(total_vol_sum)
            total_price_distance_constant[fee_rate].append(price_distance)
            print(f"Seed {seed}: Total PnL for fee rate {fee_rate}: {total_pnl_sum}")
            print(f"Seed {seed}: Total Fee for fee rate {fee_rate}: {total_fee_sum}")
            print(f"Seed {seed}: Total Volume for fee rate {fee_rate}: {total_vol_sum}")
            print(f"Seed {seed}: Total price_distance for fee rate {fee_rate}: {price_distance}")
    # Collect total PnL for RL agent
    total_pnls_rl = []
    total_fees_rl = []
    total_vols_rl = []
    total_dynamic_fee = []
    total_price_distance_rl = []

    for seed in range(iterations):
        seed = seed + int(time.time())
        total_pnl, total_fee, total_vol, dynamic_fees, price_distance = simulate_with_rl_amm(traders, seed, maker_dir)
        total_pnl_sum = sum(total_pnl.values())  # Sum the PnL of all traders
        total_fee_sum = sum(total_fee.values())  # Sum the PnL of all traders
        total_vol_sum = sum(total_vol.values())  # Sum the volume of all traders
        total_pnls_rl.append(total_pnl_sum)
        total_fees_rl.append(total_fee_sum)
        total_vols_rl.append(total_vol_sum)
        total_dynamic_fee.extend(dynamic_fees)
        total_price_distance_rl.append(price_distance)
        print(f"Seed {seed}: Total PnL for RL: {total_pnl_sum}")
        print(f"Seed {seed}: Total Fee for RL: {total_fee_sum}")
        print(f"Seed {seed}: Total Volume for RL: {total_vol_sum}")
        print(f"Seed {seed}: Dynamic Fee Rate: {np.mean(dynamic_fees)}")
        print(f"Seed {seed}: Total price_distance for RL: {price_distance}")
        
    # Save merged results to CSV
    save_merged_results_to_csv(results_dir, "total_pnls.csv", total_pnls_constant, total_pnls_rl)
    save_merged_results_to_csv(results_dir, "total_fees.csv", total_fees_constant, total_fees_rl)
    save_merged_results_to_csv(results_dir, "total_vols.csv", total_vols_constant, total_vols_rl)
    save_merged_results_to_csv(results_dir, "total_price_distance.csv", total_price_distance_constant, total_price_distance_rl)
    total_dynamic_fee = pd.DataFrame(total_dynamic_fee)
    total_dynamic_fee.to_csv(os.path.join(results_dir, "total_dynamic_fee.csv"), index=False)

    # Plot results
    plot_total_pnls(total_pnls_constant, total_pnls_rl)
    plot_total_fees(total_fees_constant, total_fees_rl)
    plot_total_vols(total_vols_constant, total_vols_rl)
    plot_total_dynamic_fee(total_dynamic_fee)
    plot_total_price_distance(total_price_distance_constant, total_price_distance_rl)
    
    
if __name__ == "__main__":
    trader_dir = '/Users/haofu/AMM-Python/models/models_trader_final'
    maker_dir = '/Users/haofu/AMM-Python/models/market_maker_final/rl_maker_40000000_steps.zip'
    main(trader_dir, maker_dir, iterations=1000)