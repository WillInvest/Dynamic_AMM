import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
import os
import torch
import sys
from tqdm import tqdm
sys.path.append('../..')

from env.multiAmm import MultiAgentAmm
from env.market import MarketSimulator
from env.new_amm import AMM

def get_highest_steps_model(models):
    # Extract the step numbers and create a tuple of (step_number, model_name)
    step_model_pairs = []
    for model in models:
        # Extract the step number from the model name
        step_number = int(model.split('steps')[-1])
        step_model_pairs.append((step_number, model))
    
    # Sort the list of tuples by step number
    step_model_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Return the model with the highest steps
    return step_model_pairs[0][1]

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
    total_gas = env.total_gas
    total_rewards = env.total_rewards
    
    return cumulative_fee, cumulative_reward, total_rewards, total_gas

def generate_fee_csv(root_path, n_agent, fee_rates, sigmas, iterations):
    
    for agent_seed in range(n_agent):
        for fee_rate in fee_rates:
            for sigma in sigmas:
                data = []
                agent_seed = int(agent_seed)
                fee_rate = round(fee_rate, 2)
                sigma = round(sigma, 1)
                model_dir = os.path.join(root_path, f"r{fee_rate:.2f}_s{sigma:.1f}")
                model_names = [f for f in os.listdir(model_dir) if f.startswith('TD3_step')]
                # Get the highest steps model
                best_model = get_highest_steps_model(model_names)
                model_path = os.path.join(model_dir, best_model)

                # load model
                model = TD3.load(path=model_path)
                for r in np.arange(0.01, 0.08, 0.01):
                    for s in np.arange(0.2, 1.0, 0.2):
                        r = round(r, 2)
                        s = round(s, 1)

                        # Setup environment and agent
                        market = MarketSimulator(start_price=1, deterministic=False, steps=500, sigma=s)
                        amm = AMM(initial_a=10000, initial_b=10000, fee=r)  # Set your fee rate
                        env = MultiAgentAmm(market, amm)
                        print(f"Calculating fee for rate ({r}), sigma ({s})")

                        for iter in tqdm(range(iterations), desc=f'model_r{fee_rate:.2f}_s{sigma:.1f}'):    
                            cumulative_fee, cumulative_reward, total_reward, total_gas = calculate_fee(model=model, env=env)
                            
                            data.append({
                                "risk_aversion": 0.2,
                                "model_fee_rate": fee_rate,
                                "model_sigma": sigma,
                                "fee_rate": r,
                                "sigma": s,
                                "iterations": iter,
                                "fee": cumulative_fee,
                                "reward": cumulative_reward,
                                "total_reward": total_reward,
                                "total_gas": total_gas 
                            })
        
                data = pd.DataFrame(data)
                csv_path = os.path.join(root_path, f"csv_file")

                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)

                # Save the DataFrame to a CSV file
                csv_name = os.path.join(csv_path, f"fee_rate_{fee_rate}_sigma_{sigma}.csv")
                data.to_csv(csv_name)
                        


if __name__ == "__main__":
    BASE_PATH = '/home/shiftpub/AMM-Python/stable_baseline/multiple_agent/models'
    MODEL_TYPE = 'TD3'
    FEE_RATES = np.arange(0.01, 0.04, 0.01)
    SIGMAS = [0.2, 0.4, 0.6, 0.8]
    N_AGENT = 1
    ITERATIONS = 30
    
    generate_fee_csv(root_path=BASE_PATH,
                         fee_rates=FEE_RATES,
                         sigmas = SIGMAS,
                         n_agent=N_AGENT,
                         iterations=ITERATIONS)               
                    