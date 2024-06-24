import sys
import os
 
sys.path.append('..')
 
from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import TD3
import torch
import random
EPSILON = 1e-5
 
class MultiAgentAmm(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 shares=10000,
                 model_path = None,
                 rule_based = True
                ) -> None:
        super().__init__()
        self.amm = amm
        self.fee_rate = amm.fee
        self.market = market
        self.shares = shares
        self.step_count = 0
        self.max_steps = 500
        self.cumulative_fee = 0
        self.total_rewards = 0
        self.total_gas = 0
        self.model_path = model_path
        # observation space = [market price, amm price, gas price]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, 1], dtype=np.float32))
        # action space = [swap percentage, gas fee]
        self.action_space = spaces.Box(low=np.array([-1., -1.]),
                                       high=np.array([1., 1.]), dtype=np.float32)
        self.done = False
        self.rule_based = rule_based
        if not rule_based:
            self.agent1 = TD3.load(self.model_path)
            print(f"agent loaded from {self.model_path}")

    def step(self, actions1: np.array) -> Tuple[np.array, float, float, bool, dict]:  
        # actions1, actions2 = actions1
        # Scale action to avoid depletion too quickly
        obs = self.get_obs()
        if self.rule_based:
            swap_rate2 = (obs[0] - obs[1]) / obs[1]
            tip_rate2 = 0.0001
        else:
            actions2, _state = self.agent1.predict(obs) 
            print(f"fix_model_action: {actions2}")
            swap_rate2, tip_rate2 = actions2[0] * 0.2, actions2[1] * 1e-4
            
        swap_rate1, tip_rate1 = actions1[0] * 0.2, actions1[1] * 1e-4
        
        def execute_trade(swap_rate, tip_rate, asset_in, asset_out):
            info = self.amm.swap(swap_rate)
            self.amm.fee = self.fee_rate   # set back to the original rate
            asset_delta = info['asset_delta']
            fee = info['fee']
            amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
            market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
            gas_fee = asset_delta[asset_in] * tip_rate * self.market.get_ask_price(asset_in) / self.shares
            # print(f"asset_in: {asset_delta[asset_in]} | asset_out: {asset_delta[asset_out]} | MB: {self.market.get_bid_price(asset_out)} | amm_cost: {amm_cost} | market_gain: {market_gain} | gas_fee: {gas_fee}")
            reward = (market_gain - amm_cost - gas_fee) / self.shares
            return reward, fee, gas_fee

        def determine_assets(swap_rate):
            if swap_rate > 0:
                return 'B', 'A'
            else:
                return 'A', 'B'

        def process_trades(swap_rate1, tip1, swap_rate2, tip2):
            rew1 = rew2 = gas1 = gas2 = 0
            fee1 = fee2 = {'A': 0, 'B': 0}  # Initialize fees to avoid reference before assignment

            if tip1 >= tip2:
                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1, gas1 = execute_trade(swap_rate1, tip1, asset_in1, asset_out1)

                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2, gas2 = execute_trade(swap_rate2, tip2, asset_in2, asset_out2)
            else:
                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2, gas2 = execute_trade(swap_rate2, tip2, asset_in2, asset_out2)

                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1, gas1 = execute_trade(swap_rate1, tip1, asset_in1, asset_out1)

            return rew1, rew2, fee1, fee2, gas1, gas2

        # process trades
        rew1, rew2, fee1, fee2, gas1, gas2 = process_trades(swap_rate1, tip_rate1, swap_rate2, tip_rate2)


        # Update cumulative fee
        self.cumulative_fee += (fee1['A'] + fee1['B'] + fee2['A'] + fee2['B'])
        # self.total_rewards += (rew1 + rew2)
        # self.total_gas += (gas1 + gas2)
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
        
        # panalize negative reward and give more weights to the current agent
        modified_rew1 = 2 * rew1 if rew1<0 else rew1
        modified_rew2 = 2 * rew2 if rew2<0 else rew2
        total_rewards = 2 * modified_rew1 + modified_rew2

        # Advance market and gas price to the next state
        self.market.next()
        next_obs = self.get_obs()

        return next_obs, rew1, self.done, False, {'total_rewards': (total_rewards),
                                                           'reward1': rew1,
                                                           'reward2': rew2,
                                                           'total_gas': (gas1 + gas2)}

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.cumulative_fee = 0
        self.total_rewards = 0
        self.total_gas = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}
 
    def get_obs(self) -> np.array:
        cur_market_price = (self.market.AP / self.market.initial_price) / (self.market.BP / self.market.initial_price)
        cur_amm_price = (self.amm.reserve_b / self.shares) / (self.amm.reserve_a / self.shares)
        return np.array([cur_market_price, cur_amm_price, 0], dtype=np.float32)
 
    def render(self, mode='human'):
        pass
 
if __name__ == "__main__":
    model_path = '/Users/haofu/AMM-Python/stable_baseline/models/TD3/agent_seed_0/fee_0.01/sigma_0.2/TD3_best_model.zip'
    market = MarketSimulator(start_price=1, deterministic=True)
    amm = AMM(initial_a=8000, initial_b=10000, fee=0.02)  # Set your fee rate
    env = MultiAgentAmm(market, amm, model_path=model_path, rule_based=True)
    
    obs, _ = env.reset()
    
    action = np.array([0.1, 0.1])
    
    for _ in range(100):
        action = env.action_space.sample()
        print(action)
        obs, rew, done, truncated, info = env.step(action)
        print(f"obs: {obs} | rew: {rew} | done: {done}")
    

    
    # for _ in range(100):
    
    #     obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cuda:0').unsqueeze(0)
    #     action1 = agent.act1.get_action(obs_tensor).detach().cpu().numpy()[0]
    #     action2 = agent.act2.get_action(obs_tensor).detach().cpu().numpy()[0]
    #     print(f"action1: {action1} | action2: {action2}")

    #     obs, rew1, rew2, done, truncated, info = env.step(action1, action2)
        
    #     print(f"obs: {obs} | rew1: {rew1} | rew2: {rew2} | done: {done}")
        
    #     if done:
    #         obs, _ = env.reset()

        
        
        