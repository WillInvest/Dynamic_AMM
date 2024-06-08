import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
sys.path.append('../..')

from env.amm_env import ArbitrageEnv
from env.market import GBMPriceSimulator
from env.new_amm import AMM

from stable_baselines3 import PPO, DDPG, TD3




def plot_and_rank_models(base_path, epsilon, fee_rate, model_type):
    plot_path = os.path.join(base_path, "saved_plot")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    model_dirs = [d for d in os.listdir(base_path) if d.startswith(model_type)]
    model_dirs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort directories by step number extracted from the name

    def load_model(algorithm, model_path, env):
        """Create the RL model based on the selected algorithm."""
        if algorithm in ("PPO", "ppo"):
            return PPO.load(model_path, env)
        elif algorithm in ("DDPG", "ddpg"):
            return DDPG.load(model_path, env)
        elif algorithm in ("TD3", "td3"):
            return TD3.load(model_path, env)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
    def plot_models(env, model, model_step, plot_path):
        
        # Simulation and plotting logic here
        state, _ = env.reset()
        rewards, amm_bid_step, amm_ask_step, amm_bids, amm_asks, market_bids, market_asks, distances  = [], [], [], [], [], [], [], []

        for _ in range(500):
            action, _state = model.predict(state, deterministic=True)         
            state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            ammAsk = state[1] * (1+fee_rate)
            ammBid = state[1] / (1+fee_rate)
            amm_bids.append(ammBid)
            amm_asks.append(ammAsk)
            ask_ratio = state[0] * (1+epsilon)
            bid_ratio = state[0] / (1+epsilon)
            market_asks.append(ask_ratio)
            market_bids.append(bid_ratio)
            distances.append(info['distance'])
             # Create stairs effect for AMM bid/ask prices
            if len(amm_bids) > 1:
                amm_bid_step.append(amm_bids[-2])
                amm_ask_step.append(amm_asks[-2])
            else:
                amm_bid_step.append(amm_bids[-1])
                amm_ask_step.append(amm_asks[-1])
        # plot
        steps = min(500, env.step_count)
        plt.figure(figsize=(20, 10))
        plt.plot(market_asks, label='Market Ask', color='red')
        plt.plot(market_bids, label='Market Bid', color='blue')
        plt.step(np.arange(steps), amm_ask_step, where='mid', label='AMM Ask', linestyle='--', color='red')
        plt.step(np.arange(steps), amm_bid_step, where='mid', label='AMM Bid', linestyle='--', color='blue')
        plt.title(f'Model Step: {model_step} - Rewards: {sum(rewards)} - Distance: {sum(distances)}')
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{plot_path}/{model_type}_{model_step}_steps.png')        
        plt.show(block=False)
        plt.pause(0.15)
        plt.close()
    
    def create_gif(source_folder, output_file, duration=5, model_type=model_type):
        # Get a list of PNG images in the source folder
        images = [img for img in os.listdir(source_folder) if img.startswith(model_type)]

        # Sort images based on the integer value found in the filename
        images.sort(key=lambda x: int(x.split('_')[1]))

        # Read and collect frames
        frames = [imageio.imread(os.path.join(source_folder, img)) for img in images]

        # Save frames as a GIF
        imageio.mimsave(output_file, frames, duration=duration, loop=1)

    for model_dir in model_dirs:
        # extract the steps from the model path
        model_step = model_dir.split('_')[1].split('.')[0]
        # filter the .zip
        model_dir = os.path.join(base_path, model_dir.split('.')[0])

        
        # Setup environment and agent
        market = GBMPriceSimulator(start_price=1, deterministic=True)
        amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)  # Set your fee rate
        env = ArbitrageEnv(market, amm)
        
        # load model
        model = load_model(algorithm=model_type, model_path=model_dir, env=env)
        
        # plot models
        plot_models(model=model, env=env, model_step=model_step, plot_path=plot_path)
        
        # create gif
        gif_path = os.path.join(plot_path, "iteration.png")
        create_gif(plot_path, gif_path, duration=200)

        
if __name__ == '__main__':
    

    
    BASE_PATH = '/home/shiftpub/AMM-Python/stable_baseline/models/PPO/2024-06-07_14-34-12'
    PLOT_PATH = os.path.join(BASE_PATH, "saved_plot")
    MODEL_TYPE = 'ppo'
    FEE_RATE = 0.02
    EPSILON = 0.01

    market = GBMPriceSimulator(start_price=1, deterministic=True, epsilon=EPSILON)
    amm = AMM(initial_a=10000, initial_b=10000, fee=FEE_RATE)  # Set your fee rate
    env = ArbitrageEnv(market, amm)
    
    plot_and_rank_models(base_path=BASE_PATH, epsilon=EPSILON, fee_rate=FEE_RATE, model_type=MODEL_TYPE)

    