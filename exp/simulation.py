import os
import sys
import numpy as np
from datetime import datetime
from itertools import product
import multiprocessing as mp
from tqdm import tqdm
import polars as pl

sys.path.append(os.path.abspath('../../Dynamic_AMM'))
from env.amm import AMM
from env.oracle import OracleSimulator
from env.trader import Arbitrager

def run_simulation(params):
    sigma, fee_rate, fee_source, seed = params
    try:
        amm = AMM(fee_source=fee_source, fee_rate=fee_rate)
        oracle = OracleSimulator(amm=amm, sigma=sigma, steps=2000, seed=seed)
        trader = Arbitrager(oracle)
        
        for _ in range(oracle.steps):
            swap_info = trader.step()
            
        result = {
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
        del amm, oracle, trader
        return result
    except Exception as e:
        print(f"Error in simulation {params}: {e}")
        return None

def main():
    # Set spawn context
    mp.set_start_method('spawn')
    
    # Setup
    root_dir = '/home/shiftpub/Dynamic_AMM/results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'{root_dir}/simulation_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Parameters
    sigmas = np.arange(0.1, 0.5, 0.1)
    fee_rates = np.arange(0.001, 0.1, 0.001)
    fee_sources = [1]
    n_seeds = 50000
    chunk_size = 2000
    n_chunks = n_seeds // chunk_size

    # Setup multiprocessing
    n_workers = int(mp.cpu_count() - 2)
    
    try:
        for chunk_id in range(n_chunks):
            start_seed = chunk_id * chunk_size
            chunk_params = list(product(
                sigmas,
                fee_rates,
                fee_sources,
                range(start_seed, start_seed + chunk_size)
            ))
            
            # Create new pool for each chunk
            with mp.Pool(processes=n_workers) as pool:
                results = list(tqdm(
                    pool.imap(run_simulation, chunk_params),
                    total=len(chunk_params),
                    desc=f"Chunk {chunk_id}/{n_chunks}"
                ))
            
            # Save results
            valid_results = [r for r in results if r is not None]
            if valid_results:
                chunk_df = pl.DataFrame(valid_results)
                chunk_df.write_parquet(
                    f'{results_dir}/chunk_{chunk_id}.parquet',
                    compression='snappy'
                )
            
            # Merge periodically
            if (chunk_id + 1) % 20 == 0:
                parquet_files = [f for f in os.listdir(results_dir) if f.endswith('.parquet')]
                merged_df = pl.concat([
                    pl.scan_parquet(f'{results_dir}/{f}')
                    for f in parquet_files
                ]).collect()
                
                merged_df.write_parquet(
                    f'{results_dir}/merged_results_{chunk_id}.parquet',
                    compression='snappy'
                )
                
                for f in parquet_files:
                    if not f.startswith('merged_'):
                        os.remove(f'{results_dir}/{f}')

    except Exception as e:
        print(f"Error in simulation: {e}")

if __name__ == '__main__':
    main()