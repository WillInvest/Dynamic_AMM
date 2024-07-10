import os
import sys
import time
import wandb
import numpy as np
import warnings
from collections import deque
import random
import matplotlib.pyplot as plt
import imageio

# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress TensorFlow warnings and informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# sys.path.append('../../..')

from stable_baseline.env.multiAmm import MultiAgentAmm
from stable_baseline.env.amm_env import ArbitrageEnv
from stable_baseline.env.market import MarketSimulator
from stable_baseline.env.new_amm import AMM

import gymnasium as gym
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

def create_gif(plot_path, output_file, duration):
    # Get a list of PNG images in the source folder
    images = [img for img in os.listdir(plot_path) if img.endswith('.png')]

    # Sort images based on the integer value found in the filename
    images.sort(key=lambda x: (float(x.split('steps_')[1].split('_')[0])))
    # Read and collect frames
    frames = [imageio.imread(os.path.join(plot_path, img)) for img in images]

    # Save frames as a GIF
    imageio.mimsave(output_file, frames, duration=duration/1000.0, loop=1)
    
    # Delete all PNG images after creating the GIF
    for img in images:
        os.remove(os.path.join(plot_path, img))


def plot_multiple(env, root_path, fig_path):
    eval_steps = 50
    fee_rate = env.amm.fee
    epsilon = env.market.epsilon
    sigma = env.market.sigma
    model_paths = [f for f in os.listdir(root_path) if f.startswith('TD3_step')]
    model_paths.sort(key=lambda x: int(x.split('steps')[-1]))
    
    for model_index in np.arange(1e5, 5e6, 1e5):
        model_index = int(model_index)
        model_name = f'TD3_step_model_r_{fee_rate:.2f}_s_{sigma:.1f}_rule_based_steps{model_index}'
        model = TD3.load(os.path.join(root_path, model_name))
        state, _ = env.reset()
        # model_index = model_path.split('steps')[-1]
        rewards, amm_bid_step, amm_ask_step, amm_bids, amm_asks, market_bids, market_asks  = [], [], [], [], [], [], []
        for _ in range(eval_steps):
            action, _state = model.predict(state, deterministic=True) 
            state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            ammAsk = state[1] * (1+fee_rate)
            ammBid = state[1] / (1+fee_rate)
            amm_bids.append(ammBid)
            amm_asks.append(ammAsk)
            ask_ratio = env.market.get_ask_price('A') / env.market.get_bid_price('B')
            bid_ratio = env.market.get_bid_price('A') / env.market.get_ask_price('B')
            market_asks.append(ask_ratio)
            market_bids.append(bid_ratio)
            
            # Create stairs effect for AMM bid/ask prices
            amm_bid_step.append(ammBid)
            amm_ask_step.append(ammAsk)
            # Create stairs effect for AMM bid/ask prices
            # if len(amm_bids) > 1:
            #     amm_bid_step.append(amm_bids[-2])
            #     amm_ask_step.append(amm_asks[-2])
            # else:
            #     amm_bid_step.append(amm_bids[-1])
            #     amm_ask_step.append(amm_asks[-1])
        # Adjust amm_bid_step and amm_ask_step to be one step to the left
        # if len(amm_bid_step) > 1:
        #     amm_bid_step = amm_bid_step[:-1]
        #     amm_ask_step = amm_ask_step[:-1]
        # plot
        steps = min(50, env.step_count)
        plt.figure(figsize=(20, 10))
        plt.plot(market_asks, label='Market Ask', color='red')
        plt.plot(market_bids, label='Market Bid', color='blue')
        plt.step(np.arange(steps), amm_ask_step, where='mid', label='AMM Ask', linestyle='--', color='red')
        plt.step(np.arange(steps), amm_bid_step, where='mid', label='AMM Bid', linestyle='--', color='blue')
        plt.title(f'Train_steps_{model_index} - Rewards: {sum(rewards)} - total_rewards: {info["total_rewards"]} - total_gas: {info["total_gas"]}')
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        fig_name = f'train_steps_{model_index}_r{fee_rate:.2f}_s{sigma}.png'
        print(f"Saving {fig_name}...")
        fig_dir = os.path.join(fig_path, fig_name)
        plt.savefig(fig_dir)
        plt.show(block=False)
        plt.pause(0.15)
        plt.close()
 
    
if __name__ == '__main__':

    for FEE_RATE in np.arange(0.05, 0.2, 0.01):
        for SIGMA in np.arange(0.2, 1.0, 0.2):
            SIGMA = round(SIGMA, 1)
            FEE_RATE = round(FEE_RATE, 2)
            ROOT_PATH = f'/home/shiftpub/AMM-Python/stable_baseline/multiple_agent/models/r{FEE_RATE:.2f}_s{SIGMA:.1f}'
            fig_path = os.path.join('/home/shiftpub/AMM-Python/stable_baseline/multiple_agent', 'figs')
            os.makedirs(fig_path, exist_ok=True)
            market = MarketSimulator(deterministic=True, steps=50, sigma=SIGMA)
            amm = AMM(fee=FEE_RATE)
            env = MultiAgentAmm(market, amm, rule_based=True)
            
            
            plot_multiple(env=env, root_path=ROOT_PATH, fig_path=fig_path)
            gif_path = os.path.join(fig_path, f"gif_r{FEE_RATE:.2f}_s{SIGMA:.1f}.gif")
            create_gif(fig_path, gif_path, duration=200)
