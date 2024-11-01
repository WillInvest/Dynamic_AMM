import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from env.oracle import OracleSimulator
from env.amm import AMM
from env.trader import Arbitrager
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the simulation function
def run_simulation_with_constant_fee_rate(fee_rate, sigma, config):
    amm = AMM(fee=fee_rate, fee_pool=config['fee_pool'])
    market = OracleSimulator(sigma=sigma,
                             mu=config['mu'],
                             spread=config['spread'],
                             dt=config['dt'],
                             start_price=config['start_price'],
                             steps=config['steps'])
    trader = Arbitrager(amm, market)
    
    # Loop over market steps
    for _ in range(int(market.steps)):
        # Get trader observations
        trader.swap()
        market.next()
    # make the amm back to the mid price of the market as the initial state
    # if amm.ls/amm.lr > market.pr/market.ps:
    #     x_r = (np.sqrt(amm.ls * amm.lr / (market.pr/market.ps)) - amm.lr) / (1-amm.f)
    # elif amm.ls/amm.lr < market.pr/market.ps:
    #     x_r = (np.sqrt(amm.ls * amm.lr / (market.pr/market.ps)) - amm.lr)
    # else:
    #     x_r = 0
    # amm.swap(x_r)
    total_pnl = trader.pnl
    total_fee = trader.total_fee
    total_volume = trader.total_number_trade
    ending_pool_value = amm.lr * market.get_mid_price('r') + amm.ls * market.get_mid_price('s')
    starting_pool_value = amm.initial_lr * market.get_mid_price('r') + amm.initial_ls * market.get_mid_price('s')
    impermanent_loss = ending_pool_value - starting_pool_value
    net_profit = total_fee + impermanent_loss
    
    return total_pnl, total_fee, total_volume, impermanent_loss, net_profit, amm.lr, amm.ls, market.get_mid_price('r'), market.get_mid_price('s')

def parallel_constant(iterations, config, sigma, results_dir):
    # Create a directory for storing results
    os.makedirs(results_dir, exist_ok=True)

    # Define the fee rates
    max_fee_rate = 1
    min_fee_rate = 0.05
    fee_rates = np.round(np.arange(min_fee_rate, max_fee_rate, min_fee_rate), 4)

    # Start parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        future_to_sim = {}
        simulation_results = []
        # Submit the tasks for parallel execution
        for fee_rate in fee_rates:
            for _ in range(iterations):
                future = executor.submit(run_simulation_with_constant_fee_rate, fee_rate, sigma, config)
                future_to_sim[future] = (sigma, fee_rate)

        # Process the results as they are completed
        for future in tqdm(as_completed(future_to_sim), total=len(future_to_sim), desc=f'Sigma {sigma} parallel execution'):
            sigma, fee_rate = future_to_sim[future]
            total_pnl, total_fee, total_vol, impermanent_loss, net_profit, amm_lr, amm_ls, mkt_r, mkt_s = future.result()
             # Store the results in a list of dictionaries
            simulation_results.append({
                'fee_rate': fee_rate,
                'sigma': sigma,
                'total_pnl': total_pnl,
                'total_fee': total_fee,
                'total_vol': total_vol,
                'impermanent_loss': impermanent_loss,
                'net_profit': net_profit,
                'amm_lr': amm_lr,
                'amm_ls': amm_ls,
                'mkt_r': mkt_r,
                'mkt_s': mkt_s
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(simulation_results)

        # Save to CSV and use the timestamp as part of the filename
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"static_simulation_results_{sigma}_{time_stamp}.csv"
        print(f"Saving results to {file_name}")
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
    
    import psutil

    def kill_processes_by_name(process_name):
        # Iterate over all running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Check if process contains the target script in its command line
                if process_name in ' '.join(proc.info['cmdline']):
                    print(f"Killing process {proc.info['pid']} with command: {proc.info['cmdline']}")
                    proc.terminate()  # Gracefully terminate the process
                    proc.wait(timeout=5)  # Wait for the process to terminate
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Handle the case where the process no longer exists or cannot be accessed
                pass

    # Call the function to kill all processes containing "parallel_dummy.py"
    # kill_processes_by_name("parallel_dummy.py")


        
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f'/home/shiftpub/Dynamic_AMM/results/dynamic_fees/{time_stamp}'
    results_dir = f"{os.path.expanduser('~')}/Dynamic_AMM/results_w25/dummy_results"

    start_price=500
    mu=0.06
    dt=1/(252*6.5*60*60)
    steps=int(60*60*6.5)
    spread=0.005
        
    
    config = {
        'mu' : mu, 
        'spread' : spread,
        'dt' : dt,
        'start_price' : start_price,
        'steps' : steps,
        'save_folder' : save_folder,
        'fee_pool' : False
    }
    
    sigma_values = np.round(np.arange(0.05, 0.425, 0.025), 3)

    print(f"Sigma values: {sigma_values}")
    for _ in range(120):
        for sigma in sigma_values:
            parallel_constant(100, config, sigma=sigma, results_dir=results_dir)
