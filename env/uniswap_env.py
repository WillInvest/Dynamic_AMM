import sys
import os
 
sys.path.append('..')
 
from env.market import MarketSimulator
from env.new_amm import AMM
from env.gas_fee import GasFeeSimulator
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
import torch
EPSILON = 1e-5
 
class UniSwapEnv(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 gas_fee: GasFeeSimulator,
                 shares=10000
                ) -> None:
        super().__init__()
        self.amm = amm
        self.market = market
        self.gas = gas_fee
        self.shares = shares
        self.step_count = 0
        self.max_steps = 500
        self.cumulative_fee = 0
        # observation space = [market price, amm price, gas price]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32))
        # action space = [swap percentage, gas fee]
        self.action_space = spaces.Box(low=np.array([-1., 0]),
                                       high=np.array([1., 1]), dtype=np.float32)
        self.done = False

    def step(self, actions1: np.array, actions2: np.array) -> Tuple[np.array, float, float, bool, dict]:  
        # actions1, actions2 = actions1
        # Scale action to avoid depletion too quickly
        swap_rate1 = actions1[0] * 0.2
        gwei1 = self.gas.min_gwei + actions1[1] * self.gas.spread
            
        swap_rate2 = actions2[0] * 0.2
        gwei2 = self.gas.min_gwei + actions2[1] * self.gas.spread
            
        def execute_trade(swap_rate, gwei, asset_in, asset_out):
            info = self.amm.swap(swap_rate)
            asset_delta = info['asset_delta']
            fee = info['fee']
            amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
            market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
            gas_cost = gwei * self.gas.price
            reward = (market_gain - amm_cost - gas_cost) / self.market.initial_price 
            return reward, fee

        def determine_assets(swap_rate):
            if swap_rate > 0:
                return 'B', 'A'
            else:
                return 'A', 'B'

        def process_trades(swap_rate1, gwei1, swap_rate2, gwei2):
            rew1 = rew2 = 0
            fee1 = fee2 = {'A': 0, 'B': 0}  # Initialize fees to avoid reference before assignment

            if gwei1 >= gwei2:
                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1 = execute_trade(swap_rate1, gwei1, asset_in1, asset_out1)

                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2 = execute_trade(swap_rate2, gwei2, asset_in2, asset_out2)
            else:
                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2 = execute_trade(swap_rate2, gwei2, asset_in2, asset_out2)

                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1 = execute_trade(swap_rate1, gwei1, asset_in1, asset_out1)

            return rew1, rew2, fee1, fee2

        # Example usage:
        rew1, rew2, fee1, fee2 = process_trades(swap_rate1, gwei1, swap_rate2, gwei2)


        # Update cumulative fee
        self.cumulative_fee += (fee1['A'] + fee1['B'] + fee2['A'] + fee2['B'])

        # Add one step count
        self.step_count += 1

        # Define terminal state
        if min(self.amm.reserve_a, self.amm.reserve_b) < 0.2 * self.shares:
            rew1 = rew2 = -1000
            self.done = True
                
        if self.step_count >= self.max_steps:
            self.done = True
        
        if self.market.shock_index == self.market.steps:
            self.done = True

        # Advance market and gas price to the next state
        self.market.next()
        self.gas.next()
        next_obs = self.get_obs()

        return next_obs, rew1, rew2, self.done, False, {}

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.cumulative_fee = 0
        self.amm.reset()
        self.market.reset()
        self.gas.reset()
        obs = self.get_obs()
        return obs, {}
 
    def get_obs(self) -> np.array:
        cur_market_price = (self.market.AP / self.market.initial_price) / (self.market.BP / self.market.initial_price)
        cur_amm_price = (self.amm.reserve_b / self.shares) / (self.amm.reserve_a / self.shares)
        return np.array([cur_market_price, cur_amm_price, self.gas.price], dtype=np.float32)
 
    def render(self, mode='human'):
        pass
 
if __name__ == "__main__":
    from exp.amm_ddpg import AgentDDPG 

    market = MarketSimulator(start_price=1, deterministic=False)
    amm = AMM(initial_a=10000, initial_b=10000, fee=0.02)  # Set your fee rate
    gas = GasFeeSimulator()
    env = UniSwapEnv(market, amm, gas)
    
    net_dim = (64, 64)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = AgentDDPG(net_dims=net_dim, state_dim=state_dim, action_dim=action_dim)
    
    # Print observation space and action space
    print(f"Observation space: {env.observation_space.shape[0]}")
    print(f"Action space: {env.action_space.shape[0]}")
    obs, _ = env.reset()
    
    print(f"random_action: {(torch.rand(2) * 2 - 1.0).unsqueeze(0)[0]}")
    
    
    # for _ in range(100):
    
    #     obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cuda:0').unsqueeze(0)
    #     action1 = agent.act1.get_action(obs_tensor).detach().cpu().numpy()[0]
    #     action2 = agent.act2.get_action(obs_tensor).detach().cpu().numpy()[0]
    #     print(f"action1: {action1} | action2: {action2}")

    #     obs, rew1, rew2, done, truncated, info = env.step(action1, action2)
        
    #     print(f"obs: {obs} | rew1: {rew1} | rew2: {rew2} | done: {done}")
        
    #     if done:
    #         obs, _ = env.reset()

        
        
        