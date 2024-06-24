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


def calculate_fee(model, env):
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

def generate_fee_csv(root_path, model_type, n_agent, fee_rates, sigmas, iterations):
    data = []
    
    for agent_seed in range(n_agent):
        for fee_rate in fee_rates:
            for sigma in sigmas:
                for iter in iterations:
                    agent_seed = int(agent_seed)
                    fee_rate = round(fee_rate, 2)
                    sigma = round(sigma, 1)
                    print(f"Calculating fee for agent seed ({agent_seed}), rate ({fee_rate}), sigma ({sigma})")
                    model_path = os.path.join(root_path, 
                                            f'agent_seed_{agent_seed}',
                                            f'fee_{fee_rate:.2f}',
                                            f'sigma_{sigma:.1f}')
                    plot_path = os.path.join(root_path, 
                                            "iteration_plot",
                                            f'agent_seed_{agent_seed}',
                                            f'fee_{fee_rate:.2f}',
                                            f'sigma_{sigma:.1f}')
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                        
                    model_dirs = [d for d in os.listdir(model_path) if d.startswith(model_type)]
                    model_dirs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort directories by step number extracted from the name

                    for model_dir in model_dirs:
                        # filter the .zip
                        model_dir = os.path.join(model_path, model_dir.split('.')[0])

                        # Setup environment and agent
                        market = MarketSimulator(start_price=1, deterministic=False, steps=500, sigma=sigma)
                        amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)  # Set your fee rate
                        env = ArbitrageEnv(market, amm)
                        
                        # load model
                        model = TD3.load(path=model_dir, env=env)
                        
                        cumulative_fee, cumulative_reward = calculate_fee(model=model, env=env)
                        
                        data.append({
                            "model_type": model_type,
                            "agent_seed": agent_seed,
                            "fee_rate": fee_rate,
                            "sigma": sigma,
                            "iterations": iter,
                            "fee": cumulative_fee,
                            "reward": cumulative_reward  
                        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(root_path, "csv_file")
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv(csv_path)
                    


if __name__ == "__main__":
    BASE_PATH = '/home/shiftpub/AMM-Python/stable_baseline/models/TD3/2024-06-10_10-08-52'
    MODEL_TYPE = 'TD3'
    FEE_RATES = np.arange(0.01, 0.21, 0.01)
    SIGMAS = [0.2, 0.4, 0.6, 0.8, 1.0]
    N_AGENT = 1
    ITERATIONS = 1000
    
    generate_fee_csv(root_path=BASE_PATH,
                         fee_rates=FEE_RATES,
                         sigmas = SIGMAS,
                         model_type=MODEL_TYPE,
                         n_agent=N_AGENT,
                         iterations=ITERATIONS)               
                    