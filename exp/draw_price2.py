from tianshou_AMM.env.AmmEnv import ArbitrageEnv
from market import GBMPriceSimulator
from new_amm import AMM
from amm_ddpg import AgentDDPG
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_distance(amm_bid, amm_ask, market_bid, market_ask):
    if amm_bid > market_ask:
        # Non-overlapping: AMM higher than market
        distance = amm_bid - market_ask
    elif amm_ask < market_bid:
        # Non-overlapping: AMM lower than market
        distance = market_bid - amm_ask
    else:
        # Overlapping
        if amm_ask < market_ask:
            distance = amm_ask - market_bid
        else:
            distance = market_ask - amm_bid
            
    return distance

def plot_bid_ask_prices(model_nums, fee_rate):
    # Setup
    net_dim = (64, 32)
    state_dim = 4
    action_dim = 1
    steps = 50
    
    # Define the basic zigzag pattern: rise to 1.5, drop to 0.5, return to 1
    rise_steps = steps // 3
    drop_steps = steps // 3
    return_steps = steps - (rise_steps + drop_steps)
    
    # Create the rise sequence from 1 to 1.5
    rise_sequence = np.linspace(1, 1.5, rise_steps)
    
    # Create the drop sequence from 1.5 to 0.5
    drop_sequence = np.linspace(1.5, 0.5, drop_steps)
    
    # Create the return sequence from 0.5 to 1
    return_sequence = np.linspace(0.5, 1, return_steps)
    
    # Concatenate the sequences to form the full zigzag pattern
    zigzag_sequence = np.concatenate((rise_sequence, drop_sequence, return_sequence))
    
    # Create the market with the zigzag sequence
    market = GBMPriceSimulator(start_price=1, deterministic=True)
    
    # Initialize AMM
    amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)
    
    # Initialize environment
    env = ArbitrageEnv(market, amm)
    
    all_data = []

    fig, axs = plt.subplots(1, 2, figsize=(14, 12))
    axs = axs.flatten()
    
    for i, model_num in enumerate(model_nums):
        # Initialize agent
        agent = AgentDDPG(net_dims=net_dim, state_dim=state_dim, action_dim=action_dim)
        
        # Load the pre-trained model
        agent.act.load_state_dict(torch.load(f'/Users/haofu/AMM/AMM-Python/src/env/AMM_DDPG_{fee_rate}_0.95/model_saves_step_{model_num}/actor_step_{model_num}.pth'))
        
        # Reset the environment to get the initial state
        state = env.reset()
        
        # Initialize lists to store rewards and states
        rewards = []
        state0_bid = []
        state0_ask = []
        state0_mid = []
        amm_bid = []
        amm_ask = []
        amm_bid_step = []
        amm_ask_step = []
        distances = []
        
        # Simulation loop
        for episode_steps in range(steps):
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
            tensor_action = agent.act(tensor_state)
            action = tensor_action.detach().cpu().numpy()[0]
            next_state, reward, done, truncated, _ = env.step(action)
            
            ammBid = state[1] / (1 + fee_rate)
            ammAsk = state[1] * (1 + fee_rate)
            marketBid = state[0] / (1 + 0.02)
            marketAsk = state[0] * (1 + 0.02)
            rewards.append(reward)
            state0_bid.append(marketBid)
            state0_ask.append(marketAsk)
            state0_mid.append(state[0])
            amm_bid.append(ammBid)
            amm_ask.append(ammAsk)
            distances.append(calculate_distance(amm_ask=ammAsk, amm_bid=ammBid, market_ask=marketAsk, market_bid=marketBid))
        
            
            all_data.append([state, action, next_state, reward, amm_bid[-1], amm_ask[-1], state0_bid[-1], state0_ask[-1]])
            state = next_state  # Update state for the next step
        
            # Create stairs effect for AMM bid/ask prices
            if len(amm_bid) > 1:
                amm_bid_step.append(amm_bid[-2])
                amm_ask_step.append(amm_ask[-2])
            else:
                amm_bid_step.append(amm_bid[-1])
                amm_ask_step.append(amm_ask[-1])
        
        print(f"Model {model_num} - cumulative rewards: {sum(rewards)}")
        print(f"Model {model_num} - cumulative distances: {sum(distances)}")

        # Flatten the rewards list
        rewards = [reward[0] for reward in rewards]
        
        # Calculate cumulative rewards
        cumulative_rewards = np.cumsum(rewards)
        
        # Plot the bid and ask prices for this model
        axs[i].plot(state0_bid, label='Market Bid', color='blue')
        axs[i].plot(state0_ask, label='Market Ask', color='red')
        axs[i].step(np.arange(steps), amm_bid, where='mid', label='AMM Bid', linestyle='--', color='blue')
        axs[i].step(np.arange(steps), amm_ask, where='mid', label='AMM Ask', linestyle='--', color='red')
        
        # Create a twin y-axis to plot the rewards as a bar plot
        ax2 = axs[i].twinx()
        ax2.bar(np.arange(steps), rewards, alpha=0.3, color='gray', label='Rewards')
        # ax2.plot(np.arange(steps), cumulative_rewards, color='green', linestyle='--', label='Cumulative Rewards Line')
        ax2.set_ylabel('Reward')
        ax2.legend(loc='lower left')
        
        axs[i].set_xlabel('Steps')
        axs[i].set_ylabel('Value')
        axs[i].set_title(f'AMM vs. Market with fee {fee_rate} (Model {model_num}) | Distance: {sum(distances):.2f} | Reward: {sum(rewards):.2f}')
        axs[i].legend(loc='upper left')
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Convert collected data to DataFrame
    df = pd.DataFrame(all_data, columns=['State', 'Action', 'Next State', 'Reward', 'AMM Bid', 'AMM Ask', 'Market Bid', 'Market Ask'])
    
    # Save DataFrame to CSV
    # df.to_csv('output_data.csv', index=False)

sequence = list(range(1024, 636928, 1024))

# Usage
s = 20480
# model_nums=[s*1, s*2, s*8, s*16]
model_nums=[29696,77824]

# model_nums=[s*20, s*24, s*28, s*32]
plot_bid_ask_prices(model_nums, fee_rate = 0.02)
