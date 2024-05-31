import sys
import os

sys.path.append('..')

from env.market import GBMPriceSimulator
from env.new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces
import torch

EPSILON = 1e-5

def parse_input(string):
    return [float(ele) for ele in string.split(" ") if ele]

class ArbitrageEnv:
    def __init__(self,
                 market: GBMPriceSimulator,
                 amm: AMM,
                 shares=10000,
                 USD=True) -> None:
        self.amm = amm
        self.market = market
        self.shares = shares
        self.epsilon = self.market.epsilon
        self.fee_rate = self.amm.fee
        self.cum_pnl = 0.
        self.USING_USD = USD
        self.step_count = 0
        self.max_steps = 500
        self.cumulative_fee = 0
        self.observation_space = spaces.Box(low=np.array([0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf], dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1 + EPSILON], dtype=np.float32),
                                       high=np.array([1 - EPSILON], dtype=np.float32), dtype=np.float32)

    def step(self, actions: np.array) -> Tuple[np.array, float, bool, dict]:  
        actions = actions * 0.1
        info = self.amm.swap(actions)
        asset_delta = info['asset_delta']
        fee = info['fee']
        if self.USING_USD:
            if actions > 0:
                asset_in, asset_out = 'B', 'A'
            else:
                asset_in, asset_out = 'A', 'B'
  
            amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
            market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
            rew = (market_gain - amm_cost) / self.market.initial_price
            self.cumulative_fee += fee[asset_in]
        else:
            amm_order_cost = asset_delta['B'] + fee['B']
            market_order_gain = (asset_delta['A'] + fee['A']) * (
                self.market.get_bid_price('B') if asset_delta['A'] < 0 else self.market.get_ask_price('B'))
            rew = - (market_order_gain + amm_order_cost)
            self.cumulative_fee += abs(fee['A'] + fee['B'])


        self.step_count += 1
        done = False
        condition = self.market.shock_index == 49
        if min(self.amm.reserve_a, self.amm.reserve_b) < 0.2 * self.shares or condition or self.step_count >= self.max_steps:
            done = True

        self.cum_pnl += rew
        self.market.next()
        next_obs = self.get_obs()
        distance = self.calculate_distance(next_obs)

        return next_obs, rew, done, False, {"distance": distance, "fee": self.cumulative_fee}

    def reset(self):
        self.cum_pnl = 0
        self.step_count = 0
        self.cumulative_fee = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}

    def get_obs(self) -> np.array:
        if self.USING_USD:
            if isinstance(self.amm.reserve_a, np.ndarray) and self.amm.reserve_a.size == 1:
                self.amm.reserve_a = self.amm.reserve_a.item()
            if isinstance(self.amm.reserve_b, np.ndarray) and self.amm.reserve_b.size == 1:
                self.amm.reserve_b = self.amm.reserve_b.item()
            cur_market_price = (self.market.AP / self.market.initial_price) / (self.market.BP / self.market.initial_price)
            cur_amm_price = (self.amm.reserve_b / self.shares) / (self.amm.reserve_a / self.shares)
        else:
            cur_market_price = self.market.BP
            cur_amm_price = self.amm.get_price()
            if isinstance(cur_market_price, np.ndarray) and cur_market_price.size == 1:
                cur_market_price = cur_market_price.item()
            if isinstance(cur_amm_price, np.ndarray) and cur_amm_price.size == 1:
                cur_amm_price = cur_amm_price.item()
        return np.array([cur_market_price, cur_amm_price], dtype=np.float32)

    def render(self, mode='human'):
        pass
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def calculate_distance(self, state):
        amm_ask = state[1] * (1+self.fee_rate)
        amm_bid = state[1] / (1+self.fee_rate)
        market_ask = state[0] * (1+self.epsilon)
        market_bid = state[0] / (1+self.epsilon)
        if amm_bid > market_ask:
            # Non-overlapping: AMM higher than market
            distance = amm_bid - market_ask
        elif amm_ask < market_bid:
            # Non-overlapping: AMM lower than market
            distance = market_bid - amm_ask
        else:
            # Overlapping
            distance = ((amm_ask - market_bid) + (market_ask - amm_bid))
            
        return distance
