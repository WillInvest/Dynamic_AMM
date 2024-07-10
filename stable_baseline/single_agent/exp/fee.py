import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
import os
import torch
import sys
sys.path.append('../..')

from env.amm_env import ArbitrageEnv
from env.market import MarketSimulator
from env.new_amm import AMM
from tqdm import tqdm


def calculate_fee(model_dir, fee_rate, sigma, seed):
    # load model
    model = TD3.load(path=model_dir)
    
    # Setup environment and agent
    market = MarketSimulator(start_price=1, deterministic=False, steps=500, sigma=sigma, seed=seed)
    amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)  # Set your fee rate
    env = ArbitrageEnv(market, amm)
                    
    # load model
    # Simulation and plotting logic here
    state, _ = env.reset()
    done = False
    cumulative_reward = 0

    while not done:
        action, _state = model.predict(state, deterministic=True)           
        state, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward        
    cumulative_fee = env.cumulative_fee
    
    return cumulative_fee, cumulative_reward

def generate_fee_csv(root_path, model_type, agent_seeds, fee_rates, sigmas, iterations):
    
    for agent_seed in agent_seeds:
        for fee_rate in fee_rates:
            for sigma in sigmas:
                data = []
                agent_seed = int(agent_seed)
                fee_rate = round(fee_rate, 2)
                sigma = round(sigma, 1)
                model_path = os.path.join(root_path, 
                                        f'agent_seed_{agent_seed}',
                                        f'fee_{fee_rate:.2f}',
                                        f'sigma_{sigma:.1f}')
                model_dirs = [d for d in os.listdir(model_path) if d.startswith(model_type)]
                model_dirs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort directories by step number extracted from the name
                best_model = model_dirs[-1]
                model_dir = os.path.join(model_path, best_model.split('.')[0])
                
                for r in np.arange(0, 0.21, 0.01):
                    for s in [0.2, 0.4, 0.6, 0.8, 1.0]:
                        for iter in tqdm(range(iterations), desc=f"model_{fee_rate}{sigma}_fee_{r}_sigma_{s}"):
                            cumulative_fee, cumulative_reward = calculate_fee(model_dir=model_dir, fee_rate=r, sigma=s, seed=iter)
                            data.append({
                                "model_type": model_type,
                                "agent_seed": agent_seed,
                                "fee_rate": r,
                                "sigma": s,
                                "iterations": iter,
                                "fee": cumulative_fee,
                                "reward": cumulative_reward  
                            })
    
                df = pd.DataFrame(data)
                csv_path = os.path.join(root_path, "csv_file_success")
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                    
                # Define the full path for the CSV file
                csv_file_path = os.path.join(csv_path, f"agent{fee_rate}_sigma_{sigma}.csv")

                # Save the DataFrame to the CSV file
                df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    BASE_PATH = '/home/shiftpub/AMM-Python/stable_baseline/single_agent/models/TD3/2024-06-10_10-08-52'
    MODEL_TYPE = 'TD3'
    FEE_RATES = np.arange(0.11, 0.41, 0.01)
    SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
    AGENT_SEEDS = [0]
    ITERATIONS = 100
    
    generate_fee_csv(root_path=BASE_PATH,
                         fee_rates=FEE_RATES,
                         sigmas = SIGMAS,
                         model_type=MODEL_TYPE,
                         agent_seeds=AGENT_SEEDS,
                         iterations=ITERATIONS)               
                    