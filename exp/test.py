
import os
import sys
from torch import load
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import glob
from tqdm import tqdm
sys.path.append('..')  # Ensure the parent directory is in the path if necessary

from env.uniswap_env import UniSwapEnv
from env.amm_env import ArbitrageEnv
from env.market import MarketSimulator
from env.gas_fee import GasFeeSimulator
from env.new_amm import AMM
from exp.amm_ddpg import AgentDDPG  

def create_gif(source_folder, output_file, duration=0.5):
    # This function sorts images based on the numerical step value in the filename
    images = [os.path.join(source_folder, img) for img in os.listdir(source_folder) if img.endswith('.png')]
    # Sort images based on the integer value found in the filename (e.g., "762000" in "model_saves_step_762000_rewards.png")
    images.sort(key=lambda x: int(x.split('_')[-2]))

    frames = [imageio.imread(img) for img in images]
    imageio.mimsave(output_file, frames, 'GIF', duration=duration)



def plot_single_agent(env, base_path, epsilon, fee_rate, net_dims, n_agent=2):
    model_dirs = [d for d in os.listdir(base_path) if d.startswith('model_saves_step_')]
    model_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort directories by step number extracted from the name
        
    for idx in range(1, n_agent+1):
        for model_dir in tqdm(model_dirs, desc=f'Plotting for rate :{fee_rate}'):
            total_steps = int(model_dir.split('_')[-1])
            # Simulation and plotting logic here
            model_path = os.path.join(base_path, model_dir, f'actor{idx}.pth')
            # Setup environment and agent
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            agent = AgentDDPG(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim)
            agent.act1.load_state_dict(torch.load(model_path))
            agent.act1.to('cuda:0')  # Move model to GPU

            state, _ = env.reset()
            done = False
            rewards, amm_bid_step, amm_ask_step, amm_bids, amm_asks, market_bids, market_asks, distances  = [], [], [], [], [], [], [], []

            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to('cuda:0').unsqueeze(0)  # Move tensor to GPU
                action = agent.act1(state_tensor).detach().cpu().numpy()[0] 
                next_state, reward, done, truncated, info = env.step(action)
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
                state = next_state
            
            # plot
            steps = min(500, env.step_count)
            plt.figure(figsize=(20, 10))
            plt.plot(market_asks, label='Market Ask', color='red')
            plt.plot(market_bids, label='Market Bid', color='blue')
            plt.step(np.arange(steps), amm_ask_step, where='mid', label='AMM Ask', linestyle='--', color='red')
            plt.step(np.arange(steps), amm_bid_step, where='mid', label='AMM Bid', linestyle='--', color='blue')
            plt.title(f'Model Step: {total_steps} - Rewards: {sum(rewards)}')
            plt.xlabel('Step')
            plt.ylabel('Ratio')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(base_path, f'saved_plot_single/{model_dir}_agent{idx}.png'))        
            plt.show(block=False)
            plt.pause(0.15)
            plt.close()
            
        # After all plots are saved, call create_gif
        print(f"Start to generate gif for {fee_rate} : {base_path}")
        plot_folder = os.path.join(base_path, 'saved_plot_single')
        gif_path = os.path.join(plot_folder, f'amm_simulation_{market_steps}_agent_{idx}.gif')
        create_gif(plot_folder, gif_path, duration=200)
                
        # Delete PNGs after creating GIF
        png_path = os.path.join(plot_folder, 'model_saves_step_*.png')
        for file_path in glob.glob(png_path):
            os.remove(file_path)


def plot_multi_agent(env, base_path, epsilon, fee_rate, net_dims):
    model_dirs = [d for d in os.listdir(base_path) if d.startswith('model_saves_step_')]
    model_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort directories by step number extracted from the name

    for model_dir in tqdm(model_dirs, desc=f'Plotting for rate :{fee_rate}'):
        total_steps = int(model_dir.split('_')[-1])
        model_path1 = os.path.join(base_path, model_dir, f'actor1.pth')
        model_path2 = os.path.join(base_path, model_dir, f'actor2.pth')

        # Setup environment and agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = AgentDDPG(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim)
        agent.act1.load_state_dict(torch.load(model_path1))
        agent.act2.load_state_dict(torch.load(model_path2))
        agent.act1.to('cuda:0')  # Move model to GPU
        agent.act2.to('cuda:0')  # Move model to GPU

        # Simulation and plotting logic here
        state, _ = env.reset()
        done = False
        rewards, amm_bid_step, amm_ask_step, amm_bids, amm_asks, market_bids, market_asks, distances  = [], [], [], [], [], [], [], []


        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to('cuda:0').unsqueeze(0)  # Move tensor to GPU
            action1 = agent.act1(state_tensor).detach().cpu().numpy()[0] 
            action2 = agent.act2(state_tensor).detach().cpu().numpy()[0]         
            next_state, reward1, reward2, done, truncated, info = env.step(action1, action2)
            rewards.append(reward1)
            rewards.append(reward2)
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
            state = next_state
        
        # plot
        steps = min(500, env.step_count)
        plt.figure(figsize=(20, 10))
        plt.plot(market_asks, label='Market Ask', color='red')
        plt.plot(market_bids, label='Market Bid', color='blue')
        plt.step(np.arange(steps), amm_ask_step, where='mid', label='AMM Ask', linestyle='--', color='red')
        plt.step(np.arange(steps), amm_bid_step, where='mid', label='AMM Bid', linestyle='--', color='blue')
        plt.title(f'Model Step: {total_steps} - Rewards: {sum(rewards)}')
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_path, f'saved_plot_multiple/{model_dir}_rewards.png'))        
        plt.show(block=False)
        plt.pause(0.15)
        plt.close()
    
    # After all plots are saved, call create_gif
    print(f"Start to generate gif for {fee_rate} : {base_path}")
    plot_folder = os.path.join(base_path, 'saved_plot_multiple')
    gif_path = os.path.join(plot_folder, f'amm_simulation_{market_steps}.gif')
    create_gif(plot_folder, gif_path, duration=200)
                
    # Delete PNGs after creating GIF
    png_path = os.path.join(plot_folder, 'model_saves_step_*.png')
    for file_path in glob.glob(png_path):
        os.remove(file_path)



if __name__ == "__main__":
    # Usage
    PLOTTING_SINGLE = True
    PLOTTING_MULTIPLE = True
    ROOT_PATH = '/home/shiftpub/AMM-Python/exp/saved_model_multiple_agents_random'
    net_dims = (256, 256)
    rates = [0.01]
    market_steps = 500

    
    for rate in rates:
        rate_path = os.path.join(ROOT_PATH, f"{rate:.2f}")
        for model_dir in os.listdir(rate_path):
            base_path = os.path.join(rate_path, model_dir)
            if PLOTTING_MULTIPLE:
                if not os.path.exists(os.path.join(base_path, "saved_plot_multiple")):
                    os.makedirs(os.path.join(base_path, "saved_plot_multiple"))
                market = MarketSimulator(start_price=1, deterministic=True, steps=market_steps)
                amm = AMM(initial_a=10000, initial_b=10000, fee=rate)  # Set your fee rate
                gas = GasFeeSimulator()
                env = UniSwapEnv(market, amm, gas)
                plot_multi_agent(env=env, base_path=base_path, epsilon=0.005, fee_rate=rate, net_dims=net_dims)
     
            if PLOTTING_SINGLE:
                if not os.path.exists(os.path.join(base_path, "saved_plot_single")):
                    os.makedirs(os.path.join(base_path, "saved_plot_single"))
                market = MarketSimulator(start_price=1, deterministic=True, steps=market_steps)
                amm = AMM(initial_a=10000, initial_b=10000, fee=rate)  # Set your fee rate
                gas = GasFeeSimulator()
                env = ArbitrageEnv(market, amm)
                plot_single_agent(env=env, base_path=base_path, epsilon=0.005, fee_rate=rate, net_dims=net_dims)
                

                

