import sys
import os
 # Get the path to the AMM-Python directory
sys.path.append(f'{os.path.expanduser("~")}/AMM-Python')
from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import TD3
import torch
import math
import random
 
class MultiAgentAmm(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 market_competition_level = 0.2
                ) -> None:
        super().__init__()
        self.amm = amm
        self.market = market
        self.step_count = 0
        self.market_competition_level = market_competition_level

        # Initialize the variables used in the step function
        self.rew1 = 0
        self.rew2 = 0
        self.fee1 = 0
        self.fee2 = 0
        self.swap_rate1 = 0
        self.swap_rate2 = 0
        self.urgent_level = 0
        self.total_rewards = 0
        self.cumulative_fee = 0
        self.cumulative_reward = 0
        self.done = False
        
        # observation space
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32))
        # action space
        self.action_space = spaces.Box(low=np.array([-0.05, 0.0]), high=np.array([0.05, 1.0]), shape=(2,), dtype=np.float32)
    
    def get_rule_base_action(self):
        obs = self.get_obs()
        market_ask, market_bid, amm_ask, amm_bid = obs[0:4]
        if amm_ask < market_bid:
            swap_rate2 = 1 - math.sqrt(self.amm.reserve_a * self.amm.reserve_b / (market_bid/(1+self.amm.fee))) / self.amm.reserve_a
            swap_rate2 *= self.market_competition_level
        elif amm_bid > market_ask:
            swap_rate2 = math.sqrt((self.amm.reserve_a*self.amm.reserve_b*market_ask*(1+self.amm.fee)))/self.amm.reserve_b - 1
            swap_rate2 *= self.market_competition_level
        else:
            swap_rate2 = 0

        return swap_rate2

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """
        urgent level determine whether agent place order or not.
        market competition level determine how much percent of arbitrage oppotunity will be taken by other traders in the market
        """
        # get the swap rate
        self.swap_rate1, self.urgent_level = action

        # process trades and update the reward and fee
        self.rew1, self.rew2, fee1, fee2 = self.process_trades()
        self.fee1 = fee1['A'] + fee1['B']
        self.fee2 = fee2['A'] + fee2['B']
        self.total_rewards = self.rew1 + self.rew2
        self.cumulative_fee += self.fee1 + self.fee2
        self.cumulative_reward += self.rew1
        self.step_count += 1

        if self.step_count == self.market.steps or min(self.amm.reserve_a, self.amm.reserve_b) < self.amm.initial_shares * 0.2:
            self.done = True

        # Advance market to the next state
        self.market.next()
        self.amm.next(random=True)
        next_obs = self.get_obs()

        infos = {
            'rew1': self.rew1,
            'rew2': self.rew2,
            'fee1': self.fee1,
            'fee2': self.fee2,
            'swap_rate1': self.swap_rate1,
            'swap_rate2': self.swap_rate2,
            'urgent_level': self.urgent_level,
            'total_rewards': self.total_rewards,
            'cumulative_fee': self.cumulative_fee,
            'cumulative_reward': self.cumulative_reward,
            'amm_fee': self.amm.fee,
            'market_sigma': self.market.sigma
        }

        return next_obs, self.rew1, self.done, False, infos

    
    def execute_trade(self, swap_rate, asset_in, asset_out):
        info = self.amm.swap(swap_rate)
        self.amm.fee = self.amm.fee  # set back to the original rate
        asset_delta = info['asset_delta']
        fee = info['fee']
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
        reward = (market_gain - amm_cost) / (100 * self.market.initial_price) if swap_rate != 0 else 0
        return reward, fee

    def process_trades(self):
        def determine_assets(swap_rate):
            if swap_rate > 0:
                return 'B', 'A'
            else:
                return 'A', 'B'
        rew1 = rew2 = 0
        fee1 = {'A': 0, 'B': 0}
        fee2 = {'A': 0, 'B': 0}

        # urgent_level is greater than the fee rate, which means the agent can accept current fee rate
        if self.urgent_level >= self.amm.fee:
                asset_in1, asset_out1 = determine_assets(self.swap_rate1)
                rew1, fee1 = self.execute_trade(self.swap_rate1, asset_in1, asset_out1)
                rew1 = rew1 * (1-self.urgent_level) if rew1 > 0 else rew1
                
                self.swap_rate2 = self.get_rule_base_action()
                asset_in2, asset_out2 = determine_assets(self.swap_rate2)
                rew2, fee2 = self.execute_trade(self.swap_rate2, asset_in2, asset_out2)
        else:
            self.swap_rate2 = self.get_rule_base_action()
            asset_in2, asset_out2 = determine_assets(self.swap_rate2)
            rew2, fee2 = self.execute_trade(self.swap_rate2, asset_in2, asset_out2)
            rew1 = 0

        return rew1, rew2, fee1, fee2

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.cumulative_fee = 0
        self.cumulative_reward = 0
        self.total_rewards = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}
 
    def get_obs(self) -> np.array:
        """
        return 5 states
        1) market ask price
        2) market bid price
        3) amm ask price
        4) amm bid price
        5) amm fee rate
        """
        return np.array([
            self.market.get_ask_price('A') / self.market.get_bid_price('B'),
            self.market.get_bid_price('A') / self.market.get_ask_price('B'),
            (self.amm.reserve_b / self.amm.reserve_a) * (1+self.amm.fee),
            (self.amm.reserve_b / self.amm.reserve_a) / (1+self.amm.fee),
            self.amm.fee
            ], dtype=np.float32)
 
    def render(self, mode='human'):
        pass
 
if __name__ == "__main__":
    market = MarketSimulator()
    amm = AMM()  # Set your fee rate
    env = MultiAgentAmm(market, amm)
    # sample_action = env.action_space.sample()
    sample_action = np.array([1, 0.5])
    obs, _ = env.reset()
    print(f"action: {sample_action}")
    print(f"obs: {obs}")
    print(amm.fee)
    obs, reward, done, truncated, info = env.step(sample_action)
    
    print(f"reward: {reward}")
    

        
        
        