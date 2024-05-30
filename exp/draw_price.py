from tianshou_AMM.env.AmmEnv import ArbitrageEnv
from market import GBMPriceSimulator
from new_amm import AMM
from amm_ddpg import AgentDDPG
import torch
import matplotlib.pyplot as plt
import numpy as np

# Setup
net_dim = (64, 32)
state_dim = 4
action_dim = 1

# Number of steps in the sequence
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

print(f"zigzag_sequence: {zigzag_sequence}")

# Create the market with the zigzag sequence
market = GBMPriceSimulator(start_price=1, deterministic=True, shocks=zigzag_sequence)

# Initialize AMM, environment, and agent
amm = AMM(initial_a=10000, initial_b=10000, fee=0.02)
env = ArbitrageEnv(market, amm)
agent = AgentDDPG(net_dims=net_dim, state_dim=state_dim, action_dim=action_dim)

# Load the pre-trained model
model_num = 300032
agent.act.load_state_dict(torch.load(f'/Users/haofu/AMM/AMM-Python/src/env/AMM_DDPG/model_saves_step_{model_num}/actor_step_{model_num}.pth'))

# Reset the environment to get the initial state
state = env.reset()
print(f"state: {state}")

# Initialize lists to store rewards and states
rewards = []
state0_bid = []
state0_ask = []
state0_mid = []
amm_bid = []
amm_ask = []
amm_bid_step = []
amm_ask_step = []

# Simulation loop
for episode_steps in range(50):
    tensor_state = torch.as_tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
    tensor_action = agent.act(tensor_state)
    action = tensor_action.detach().cpu().numpy()[0]  # No need to detach() because using torch.no_grad() outside
    print("_" * 10)
    print(f"episode_steps: {episode_steps}\n")
    state, reward, done, truncated, _ = env.step(action)
    
    print(f"reward: {reward} | action: {action}")
    
    rewards.append(reward)
    state0_bid.append(state[0] / (1 + 0.02))
    state0_ask.append(state[0] * (1 + 0.02))
    state0_mid.append(state[0])
    amm_bid.append(state[1] / (1 + 0.02))
    amm_ask.append(state[1] * (1 + 0.02))

    # Create stairs effect for AMM bid/ask prices
    if len(amm_bid) > 1:
        amm_bid_step.append(amm_bid[-2])
        amm_ask_step.append(amm_ask[-2])
    else:
        amm_bid_step.append(amm_bid[-1])
        amm_ask_step.append(amm_ask[-1])
        
# Flatten the rewards list
rewards = [reward[0] for reward in rewards]

# Calculate cumulative rewards
cumulative_rewards = np.cumsum(rewards)
print(f"cumulative reward: {cumulative_rewards}")

# Plot the bid and ask prices
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot the bid and ask prices on the left y-axis
ax1.plot(state0_bid, label='Market Bid', color='blue')
ax1.plot(state0_ask, label='Market Ask', color='red')
ax1.step(np.arange(50), amm_bid, where='mid', label='AMM Bid', linestyle='--', color='blue')
ax1.step(np.arange(50), amm_ask, where='mid', label='AMM Ask', linestyle='--', color='red')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a twin y-axis to plot the rewards as a bar plot
ax2 = ax1.twinx()
ax2.bar(np.arange(50), rewards, alpha=0.3, color='gray', label='Rewards')
ax2.plot(np.arange(50), cumulative_rewards, color='green', linestyle='--', label='Cumulative Rewards')
ax2.set_ylabel('Reward')
ax2.legend(loc='upper right')

plt.title('Zigzag Line Pattern with Bid and Ask Prices, Reward Bar Plot, and Cumulative Rewards')
plt.show()