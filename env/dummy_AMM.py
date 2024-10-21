import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import PPO, TD3
import math
 
 
class DummyAMM(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM
                ) -> None:
        super().__init__()

        self.amm = amm
        self.market = market
        self.step_count = 0
        self.done = False
        self.max_steps = 5000
        self.cumulative_pnl = 0
        self.cumulative_fee = 0

        # observation space
        low = np.zeros(6)
        high = np.inf * np.ones(6)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # action space
        self.action_space = spaces.Box(low=0.0005, high=0.0035, shape=(1,), dtype=np.float32)
        
    def get_rule_base_action(self):
        obs = self.get_trader_obs()
        market_ask, market_bid, amm_ask, amm_bid = obs[0:4]
        if amm_ask < market_bid:
            swap_rate = 1 - math.sqrt(self.amm.reserve_a * self.amm.reserve_b / (market_bid/(1+self.amm.fee))) / self.amm.reserve_a
        elif amm_bid > market_ask:
            swap_rate = math.sqrt((self.amm.reserve_a*self.amm.reserve_b*market_ask*(1+self.amm.fee)))/self.amm.reserve_b - 1
        else:
            swap_rate = 0

        return swap_rate
    
    def get_trader_obs(self) -> np.array:
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
        
    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """
        urgent level determine whether agent place order or not.
        market competence determine how much percent of arbitrage oppotunity will be taken by other traders in the market
        """
        self.amm.fee = action[0]
        assert self.amm.fee > 0, "Fee rate must be greater than 0"
        swap_rate = self.get_rule_base_action()
        info = self.amm.swap(swap_rate)
        asset_delta = info['asset_delta']
        fee = info['fee']
        asset_in, asset_out = ('A', 'B') if swap_rate < 0 else ('B', 'A')
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
        pnl = (market_gain - amm_cost) / self.market.initial_price if swap_rate != 0 else 0
        fee = fee[asset_in] * self.market.get_ask_price(asset_in) / self.market.initial_price
        self.cumulative_fee += fee
        self.cumulative_pnl += pnl
        reward = fee
        
        info = {
            'cumulative_fee' : self.cumulative_fee,
            'cumulative_pnl' : self.cumulative_pnl,
            'fee_rate' : self.amm.fee
        }
        
        # increase the step count
        self.step_count += 1
        if self.step_count == self.market.steps or min(self.amm.reserve_a, self.amm.reserve_b) < self.amm.initial_shares * 0.2:
            self.done = True
        # Advance market to the next state
        self.market.next()
        next_obs = self.get_obs()
        
        return next_obs, reward, self.done, False, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        self.cumulative_fee = 0
        self.cumulative_pnl = 0
        return obs, {}
    
    def get_obs(self) -> np.array:
        market_ask = self.market.get_ask_price('A') / self.market.get_bid_price('B')
        market_bid = self.market.get_bid_price('A') / self.market.get_ask_price('B')
        market_mid = (market_ask + market_bid) / 2
        amm_mid = self.amm.reserve_b / self.amm.reserve_a
        distance = abs(market_mid - amm_mid)
        obs = np.array([
                  self.market.get_ask_price('A') / self.market.get_bid_price('B'),
                  self.market.get_bid_price('A') / self.market.get_ask_price('B'),
                  self.amm.reserve_b / self.amm.initial_shares,
                  self.amm.reserve_a / self.amm.initial_shares,
                  self.amm.reserve_b / self.amm.reserve_a,
                  self.step_count / self.market.steps
                  ], dtype=np.float32)
        return obs
 
    def render(self, mode='human'):
        pass
 
if __name__ == '__main__':
    market = MarketSimulator(sigma=-1)
    amm = AMM()
    env = DummyAMM(market=market, amm=amm)
    env.reset()
    for _ in range(1000):
        action = [0.005]
        obs, reward, done, trauncated, info = env.step(action)
        # print(info['cumulative_fee'])