import os
import sys
sys.path.append(os.path.abspath('../../Dynamic_AMM'))
from env.amm import AMM
from env.oracle import OracleSimulator
from env.trader import Arbitrager

import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime

def run_simulation(params):
    sigma, fee_rate, fee_source, seed = params
    amm = AMM(fee_source=fee_source, fee_rate=fee_rate)
    oracle = OracleSimulator(amm=amm, sigma=sigma, steps=2000, seed=seed)
    trader = Arbitrager(oracle)
    
    for _ in range(oracle.steps):
        swap_info = trader.step()
    
    return {
        'sigma': sigma,
        'fee_rate': fee_rate,
        'fee_source': 'in' if fee_source == 1 else 'out',
        'seed': seed,
        'total_fee_dollar_value': swap_info['total_fee_dollar_value'],
        'trader_total_pnl': swap_info['trader_total_pnl'],
        'impermanent_loss': swap_info['impermanent_loss'],
        'net_profit': swap_info['net_profit'],
        'total_number_trade': swap_info['total_number_trade'],
        'account_profit': swap_info['account_profit']
    }

if __name__ == '__main__':
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'simulation_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    sigmas = np.arange(0.1, 0.5, 0.1)
    fee_rates = np.arange(0.0005, 0.0105, 0.0005)  
    fee_sources = [1, -1]
    seeds = range(1, 1001)
   
    params = list(product(sigmas, fee_rates, fee_sources, seeds))
    total_sims = len(params)
    results = []
    batch_size = 10000
   
    try:
        for i in range(0, total_sims, batch_size):
            batch_params = params[i:i+batch_size]
            with Pool() as pool:
                batch_results = list(tqdm(
                    pool.imap(run_simulation, batch_params),
                    total=len(batch_params),
                    desc=f"Batch {i//batch_size + 1}/{(total_sims-1)//batch_size + 1}"
                ))
            results.extend(batch_results)
            pd.DataFrame(results).to_csv(f'{results_dir}/intermediate_{i}.csv', index=False)

        final_df = pd.DataFrame(results)
        final_df.to_csv(f'{results_dir}/final_results.csv', index=False)
       
        # Clean up intermediate files
        for i in range(0, total_sims, batch_size):
            intermediate_file = f'{results_dir}/intermediate_{i}.csv'
            if os.path.exists(intermediate_file):
                os.remove(intermediate_file)
               
    except Exception as e:
        print(f"Error occurred: {e}")