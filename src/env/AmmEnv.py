import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# sys.path.append("..")

from src.env.market import GBMPriceSimulator
from src.amm.amm import AMM, SimpleFeeAMM
from src.amm.fee import PercentFee
# from amm.utils import parse_input
from typing import Tuple
import numpy as np
from gymnasium import spaces

EPSILON = 1e-5

def parse_input(string):
    return [float(ele) for ele in string.split(" ") if ele]



class ArbitrageEnv:
    def __init__(self,
                 market: GBMPriceSimulator,
                 amm: AMM,
                 shares=10000
                 ) -> None:
        self.amm = amm
        self.market = market
        self.shares = shares
        self.cum_pnl = 0.
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1 + EPSILON, 0], dtype=np.float32),
                                                           high=np.array([1 - EPSILON, 1]), dtype=np.float32)

    # action: [trade_size_fraction, trade_decision (take opportunity when >0.5)]

    def step(self, action: np.array) -> Tuple[
        np.array, float, bool, bool, dict]:  # next_obs, rew, done, truncated, info
        trade_size_fraction, trade_prob = action
        trade_size_fraction *= 0.2
        # print(f"trade_size_fraction: {trade_size_fraction}, trade_prob: {trade_prob}")
        # Transform the second action component to the range [0, 1]
        # trade_prob = (action[1] + 1) / 2  # Mapping from [-1, 1] to [0, 1]
        if trade_prob > 0.5:
            if trade_size_fraction > 0:
                asset_in, asset_out = 'B', 'A'
            else:
                asset_in, asset_out = 'A', 'B'
            size = abs(trade_size_fraction) * self.amm.portfolio[asset_out]
            # print(f"asset_in: {asset_in}, asset_out: {asset_out}, size: {size}")
            success, info = self.amm.trade_swap(asset_in, asset_out, -size)  # take out -size shares of outing asset

            # calculate the reward
            asset_delta = info['asset_delta']
            # print(asset_delta)
            fee = info['fee']
            amm_order_cost = asset_delta['B'] + fee['B']  # unit is always in B
            market_order_gain = (asset_delta['A'] + fee['A']) * (
                self.market.get_bid_price() if asset_delta['A'] < 0 else self.market.get_ask_price())
            rew = - (market_order_gain + amm_order_cost)
            print(f"asset_delta['A'] : {asset_delta['A']}, fee['A']: {fee['A']} | "
                  f"asset_A : {self.amm.portfolio['A']} | asset_B : {self.amm.portfolio['B']} | ")
            print(f"market_order_gain: {market_order_gain} | "
                  f"amm_order_cost: {amm_order_cost} | "
                  f"reward: {rew}")
        else:
            success, info = True, {}
            rew = 0.

        done = False
        if min(self.amm.portfolio['A'], self.amm.portfolio['B']) < 0.2 * self.shares:
            done = True

        self.cum_pnl += rew
        self.market.next()

        next_obs = self.get_obs()

        return next_obs, rew, done, not success, {}

    def reset(self):
        self.cum_pnl = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}

    def get_obs(self) -> np.array:
        cur_market_price = self.market.current_price
        tmp = self.amm.portfolio
        cur_amm_price = tmp['B'] / tmp['A']

        return np.array([cur_market_price, cur_amm_price, tmp['A']/10000, tmp['B']/10000], dtype=np.float32)

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    market = GBMPriceSimulator()
    amm_no_fee = SimpleFeeAMM(
        utility_func="constant_product",
        init_portfolio={'A': 10000, 'B': 10000, 'L': 10000},
        fee_structure=PercentFee(0.0)
    )
    env = ArbitrageEnv(market, amm_no_fee)

    # Reset the environment
    obs = env.reset()
    print(f"Initial observation: {obs}")

    # Perform a few steps with sample actions
    for step in range(10):
        # Generate a random action
        trade_size_fraction = np.random.uniform(-0.1, 0.1)  # Random number between -1 and 1
        trade_decision = np.random.uniform(0, 1)  # Random number between 0 and 1
        # action = np.array([trade_size_fraction, trade_decision], dtype=np.float32)
        # action[1] = 1
        s2string = input("Input string (i.e. A B 1): ")
        if s2string == 'r':
            env.reset()
            continue  # reset
        action = parse_input(s2string)
        print(f"\n--------------------"
              f"\nStep {step + 1}:")
        print(f"action: {action}")
        print(f"market_ask: {market.get_ask_price()} | market_bid: {market.get_bid_price()}")
        print(f"current_observation: {env.get_obs()}")
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"Next observation: {next_obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        if done:
            env.reset()
        print(env.amm)
