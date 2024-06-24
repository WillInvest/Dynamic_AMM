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

sys.path.append('../../..')

from env.multiAmm import MultiAgentAmm
from env.amm_env import ArbitrageEnv
from env.market import MarketSimulator
from env.new_amm import AMM

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
    images.sort(key=lambda x: (float(x.split('r')[1].split('_')[0]), float(x.split('s')[1].split('.png')[0])))

    # Read and collect frames
    frames = [imageio.imread(os.path.join(plot_path, img)) for img in images]

    # Save frames as a GIF
    imageio.mimsave(output_file, frames, duration=duration, loop=1)
    
    # Delete all PNG images after creating the GIF
    for img in images:
        os.remove(os.path.join(plot_path, img))


def plot_multiple(env, model_path, fig_path):
    eval_steps = 500
    fee_rate = env.amm.fee
    epsilon = env.market.epsilon
    sigma = env.market.sigma
    rewards, amm_bid_step, amm_ask_step, amm_bids, amm_asks, market_bids, market_asks  = [], [], [], [], [], [], []
    fix_model_index = env.model_path.split('_')[-1]
    model_index = model_path.split('_')[-1]
    model = TD3.load(model_path)
    state, _ = env.reset()
    for _ in range(eval_steps):
        action, _state = model.predict(state, deterministic=True) 
        print(f"model_action: {action}")        
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
    plt.title(f'Fix-{fix_model_index}-Model-{model_index} - Rewards: {sum(rewards)} - total_rewards: {info["total_rewards"]} - total_gas: {info["total_gas"]}')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    fig_name = f'f{fix_model_index}_m{model_index}_r{fee_rate:.2f}_s{sigma}.png'.format(fee_rate, sigma)
    fig_path = os.path.join(fig_path, fig_name)
    plt.savefig(fig_path)
    plt.show(block=False)
    plt.pause(0.15)
    plt.close()
 
    
if __name__ == '__main__':
    for fix_model_index in [6, 7, 8, 9]:
        for model_index in [6, 7, 8, 9]:
            FIX_PATH = f'/Users/haofu/AMM-Python/stable_baseline/models/TD3_best_model_r_0.01_s_0.2_ex_{fix_model_index}'
            MODEL_PATH = f'/Users/haofu/AMM-Python/stable_baseline/models/TD3_best_model_r_0.01_s_0.2_ex_{model_index}'
            plot_exist = False
            fig_path = f'f{fix_model_index}_m{model_index}'
            os.makedirs(fig_path, exist_ok=True)
            # Load the environment
            if not plot_exist:
                for FEE_RATE in np.arange(0.01, 0.2, 0.01):
                    FEE_RATE = round(FEE_RATE, 2)
                    market = MarketSimulator(deterministic=True)
                    amm = AMM(fee=FEE_RATE)
                    env = MultiAgentAmm(market, amm, rule_based=False, model_path=FIX_PATH)
                    # Plot the results
                    plot_multiple(env, MODEL_PATH, fig_path)
            gif_path = os.path.join(fig_path, f"gif_f{fix_model_index}_m{model_index}_r{FEE_RATE:.2f}.gif")
            create_gif(fig_path, gif_path, duration=200)
