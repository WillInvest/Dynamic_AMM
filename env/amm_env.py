import sys
import os

sys.path.append('..')

from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
import torch

EPSILON = 1e-5

class ArbitrageEnv(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 shares=10000
                ) -> None:
        super().__init__()
        self.amm = amm
        self.market = market
        self.shares = shares
        self.epsilon = self.market.epsilon
        self.fee_rate = self.amm.fee
        self.cum_pnl = 0.
        self.step_count = 0
        self.max_steps = 500
        self.cumulative_fee = 0
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1., 0.], dtype=np.float32),
                                       high=np.array([1., 1.], dtype=np.float32), dtype=np.float32)
        self.done = False

    def step(self, actions: np.array) -> Tuple[np.array, float, bool, dict]:  
        # scale action to avoid depletion too quickly
        actions = actions[0] * 0.2
        # swap tokens in amm
        info = self.amm.swap(actions)
        asset_delta = info['asset_delta']
        fee = info['fee']
        
        # decide which token should be taken out
        if actions > 0:
            asset_in, asset_out = 'B', 'A'
        else:
            asset_in, asset_out = 'A', 'B'

        # calculate reward
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
        rew = (market_gain - amm_cost) / self.market.initial_price
        self.cumulative_fee += fee[asset_in]

        # add one step count
        self.step_count += 1
        
        # define terminal state
        if min(self.amm.reserve_a, self.amm.reserve_b) < 0.2 * self.shares:
            rew = -1000
            self.done = True
            
        if self.step_count >= self.max_steps:
            self.done = True
            
        if self.market.shock_index >= self.market.steps:
            self.done = True

        self.cum_pnl += rew
        self.market.next()
        next_obs = self.get_obs()
        distance = self.calculate_distance(next_obs)

        return next_obs, rew, self.done, False, {"distance": distance, "fee": self.cumulative_fee}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.cum_pnl = 0
        self.step_count = 0
        self.cumulative_fee = 0
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

if __name__ == "__main__":
    market = MarketSimulator(start_price=1, deterministic=False)
    amm = AMM(initial_a=10000, initial_b=10000, fee=0.02)  # Set your fee rate
    env = ArbitrageEnv(market, amm, USD=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    cumulative_reward = 0
    actions = np.random.rand(100) * 0.2 - 0.1
    print(f"actions: {actions}")

    for action in actions:
        new_state, rew, done, truncated, info = env.step(action)
        cumulative_reward += rew
        print(f"cumulative_fee: {env.cumulative_fee} | type: {env.cumulative_fee.dtype}")
        