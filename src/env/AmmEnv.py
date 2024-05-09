import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market import GBMPriceSimulator
from amm.amm import AMM, SimpleFeeAMM
from typing import Tuple
import numpy as np
from gym.spaces import Discrete, Box
from amm.fee import PercentFee

EPSILON = 1e-5


class ArbitrageEnv:
    def __init__(self,
                 market: GBMPriceSimulator,
                 amm: AMM
                 ) -> None:
        self.amm = amm
        self.market = market
        self.cum_pnl = 0.
        self.observation_space = Box(low=np.array([0., 0., 0., 0.], dtype=np.float32), high=np.array(
            [np.inf, np.inf, np.inf, np.inf], dtype=np.float32))  # obs: [market price, amm price, amm inv A, amm inv B]
        self.action_space = self.action_space = Box(low=np.array([-1 + EPSILON, 0], dtype=np.float32),
                                                    high=np.array([1 - EPSILON, 1]), dtype=np.float32)
        # action: [trade_size_fraction, trade_decision (take opportunity when >0.5)]

    def step(self, action: np.array) -> Tuple[
        np.array, float, bool, bool, dict]:  # next_obs, rew, done, truncated, info
        trade_size_fraction, trade_prob = action
        if trade_prob > 0.5:
            if trade_size_fraction > 0:
                asset_in, asset_out = 'B', 'A'
            else:
                asset_in, asset_out = 'A', 'B'
            size = abs(trade_size_fraction) * self.amm.portfolio[asset_out]
            print(f"asset_in: {asset_in}, asset_out: {asset_out}, size: {size}")
            success, info = self.amm.trade_swap(asset_in, asset_out, -size)  # take out -size shares of outing asset

            # calculate the reward
            asset_delta = info['asset_delta']
            fee = info['fee']
            print(f"asset_delta: {asset_delta}, fee: {fee}")
            print(self.amm)
            amm_order_cost = asset_delta[asset_in] + fee[asset_in]  # unit is always in B
            if asset_out == 'B':
                market_order_gain = abs(asset_delta[asset_out]) * self.market.current_price
            else:
                market_order_gain = abs(asset_delta[asset_out]) / self.market.current_price
            rew = market_order_gain - amm_order_cost
            print(f"market_order_gain: {market_order_gain} | "
                  f"amm_order_cost: {amm_order_cost} | "
                  f"reward: {rew}")
        else:
            success, info = True, {}
            rew = 0.

        self.cum_pnl += rew
        self.market.next()

        next_obs = self.get_obs()

        return next_obs, rew, False, not success, {'amm_trade_info': info}

    def reset(self):
        self.cum_pnl = 0
        self.amm.reset()
        self.market.reset()
        return self.get_obs()

    def get_obs(self) -> np.array:
        cur_market_price = self.market.current_price
        tmp = self.amm.portfolio
        cur_amm_price = tmp['A'] / tmp['B']

        return np.array([cur_market_price, cur_amm_price, tmp['A'], tmp['B']])


if __name__ == '__main__':
    market = GBMPriceSimulator()
    amm_no_fee = SimpleFeeAMM(
        utility_func="constant_product",
        init_portfolio={'A': 1000, 'B': 1000, 'L': 1000},
        fee_structure=PercentFee(0.0)
    )
    env = ArbitrageEnv(market, amm_no_fee)

    # Reset the environment
    obs = env.reset()
    print(f"Initial observation: {obs}")

    # Perform a few steps with sample actions
    for step in range(5):
        # Generate a random action
        trade_size_fraction = np.random.uniform(-0.1, 0.1)  # Random number between -1 and 1
        trade_decision = np.random.uniform(0, 1)  # Random number between 0 and 1
        action = np.array([trade_size_fraction, trade_decision], dtype=np.float32)
        print(f"\n--------------------"
              f"\nStep {step + 1}:")
        print(f"action: {action}")
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"Next observation: {next_obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
