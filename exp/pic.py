
import os
import sys
sys.path.append('..')

from env.amm_env import ArbitrageEnv
from env.market import GBMPriceSimulator
from env.new_amm import AMM
from amm_ddpg import AgentDDPG
import torch
import matplotlib
# matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import time

import numpy as np

def calculate_distance(amm_bid, amm_ask, market_bid, market_ask):
    if amm_bid > market_ask:
        # Non-overlapping: AMM higher than market
        distance = amm_bid - market_ask
    elif amm_ask < market_bid:
        # Non-overlapping: AMM lower than market
        distance = market_bid - amm_ask
    else:
        # Overlapping
        distance = ((amm_ask - market_bid) + (market_ask - amm_bid))/2
            
    return distance

def plot_and_rank_models(model_nums, path, epsilon, USD=False):
    # Setup
    net_dim = (64, 32)
    state_dim = 2
    action_dim = 1
    steps = 50
    
  
    # Create the market with the zigzag sequence
    market = GBMPriceSimulator(start_price=1, deterministic=True, random=False)
    
    # Initialize AMM
    fee_rate = 0.02
    amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)
    
    # Initialize environment
    env = ArbitrageEnv(market, amm, USD=USD)
    
    # List to store model rewards
    model_rewards = []
    model_distances = []
    
    # Get current time in seconds since the epoch, then format it into a more readable form
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    # Create folder name with timestamp
    folder_path = f'saved_plot/{path}_plot'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for model_num in model_nums:
        # Initialize agent
        agent = AgentDDPG(net_dims=net_dim, state_dim=state_dim, action_dim=action_dim)
        
        # Load the pre-trained model
        agent.act.load_state_dict(torch.load(f'/Users/haofu/AMM/tianshou_AMM/exp/log/Ant-v4/ddpg/0/{path}/policy.pth'))

        
        # Reset the environment to get the initial state
        state = env.reset()
        
        # Initialize lists to store rewards and states
        rewards = []
        amm_bid_step = []
        amm_ask_step = []
        distances = []
        amm_bids = []
        amm_asks = []
        market_bids = []
        market_asks = []


        # Simulation loop
        for episode_steps in range(steps):
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
            tensor_action = agent.act(tensor_state)
            action = tensor_action.detach().cpu().numpy()[0]
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if USD:
                ammMid = state[1] #/ state[0]
                ammBid = ammMid / (1 + fee_rate)
                ammAsk = ammMid * (1 + fee_rate)
                amm_bids.append(ammBid)
                amm_asks.append(ammAsk)
                # askA = state[2] * (1 + 2 * epsilon)
                # askB = state[3] * (1 + epsilon)
                # bidA = state[2] / (1 + 2 * epsilon)
                # bidB = state[3] / (1 + epsilon)
                # ask_ratio = askA / bidB
                # bid_ratio = bidA / askB
                ask_ratio = state[0] * (1+epsilon)
                bid_ratio = state[0] / (1+epsilon)
                market_asks.append(ask_ratio)
                market_bids.append(bid_ratio)
            # **********************************************
            else:
                ammAsk = state[1] * (1+fee_rate)
                ammBid = state[1] / (1+fee_rate)
                amm_bids.append(ammBid)
                amm_asks.append(ammAsk)
                ask_ratio = state[0] * (1+epsilon)
                bid_ratio = state[0] / (1+epsilon)
                market_asks.append(ask_ratio)
                market_bids.append(bid_ratio)
            # ********~**************************************

            distances.append(calculate_distance(amm_ask=ammAsk, amm_bid=ammBid, market_ask=ask_ratio, market_bid=bid_ratio))
        
            # Create stairs effect for AMM bid/ask prices
            if len(amm_bids) > 1:
                amm_bid_step.append(amm_bids[-2])
                amm_ask_step.append(amm_asks[-2])
            else:
                amm_bid_step.append(amm_bids[-1])
                amm_ask_step.append(amm_asks[-1])
        
        cumulative_reward = sum(rewards)
        cumulative_distance = sum(distances)
        model_rewards.append((model_num, cumulative_reward))
        model_distances.append((model_num, cumulative_distance, cumulative_reward))

        print(f"Model {model_num} - cumulative rewards: {cumulative_reward}")
        print(f"Model {model_num} - cumulative distances: {cumulative_distance}")

        # Plot the bid and ask prices for this model
        plt.figure(figsize=(20, 10))
        plt.plot(market_asks, label='Market Ask', color='red')
        plt.plot(market_bids, label='Market Bid', color='blue')
        plt.step(np.arange(steps), amm_ask_step, where='mid', label='AMM Ask', linestyle='--', color='red')
        plt.step(np.arange(steps), amm_bid_step, where='mid', label='AMM Bid', linestyle='--', color='blue')
        plt.title(f'Bid and Ask Ratios --- training steps: {model_num}')
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{folder_path}/plot_model_{model_num}.png')
        plt.show(block=False)
        plt.pause(0.15)
        plt.close()
        
    
    # Rank models by cumulative rewards
    # model_rewards.sort(key=lambda x: x[1], reverse=True)
    print("\nRanking of models by cumulative rewards:")
    for rank, (model_num, cumulative_reward) in enumerate(model_rewards, 1):
        print(f"Rank {rank}: Model {model_num} - Cumulative Reward: {cumulative_reward}")
        
    # Rank models by cumulative distances
    # model_rewards.sort(key=lambda x: x[1], reverse=True)
    # print("\nRanking of models by cumulative distance:")
    # for rank, (model_num, cumulative_distance, cumulative_reward) in enumerate(model_rewards, 1):
    #     print(f"Rank {rank}: Model {model_num} - Cumulative distance: {cumulative_distance} - Cumulative reward: {cumulative_reward}")

sequence = list(range(1024, 148480, 1024))

# Usage: Plot a subset of model numbers and rank them by cumulative rewards

path = '240529-215254'
plot_and_rank_models(model_nums=sequence, path=path, epsilon=0.005, USD=False)
