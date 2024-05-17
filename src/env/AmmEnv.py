import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market import GBMPriceSimulator
# from new_amm import AMM
from amm.amm import AMM, SimpleFeeAMM
from amm.fee import PercentFee
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
        self.action_space = spaces.Box(low=np.array([-1 + EPSILON], dtype=np.float32),
                                       high=np.array([1 - EPSILON], dtype=np.float32), dtype=np.float32)

    # action: [trade_size_fraction, trade_decision (take opportunity when >0.5)]

    def step(self, actions: np.array) -> Tuple[
        np.array, float, bool, bool, dict]:  # next_obs, rew, done, truncated, info
        trade_prob = 1
        actions = actions * 0.1
        if abs(trade_prob) > 0.5:
            info = self.amm.swap(actions)
            # calculate the reward
            asset_delta = info['asset_delta']
            # print(asset_delta)
            fee = info['fee']
            # print(f"fee: {fee} | asset_delta : {asset_delta}")

            amm_order_cost = asset_delta['B'] + fee['B']  # unit is always in B
            market_order_gain = (asset_delta['A'] + fee['A']) * (
                self.market.get_bid_price() if asset_delta['A'] < 0 else self.market.get_ask_price())
            rew = - (market_order_gain + amm_order_cost)

            #
            # print(f"asset_delta['A'] : {asset_delta['A']}, fee['FA']: {fee['A']} | "
            #       f"asset_delta['B']: {asset_delta['B']} | "
            #       f"asset_A : {self.amm.reserve_a} | asset_B : {self.amm.reserve_b} | ")
            # print(f"market_order_gain: {market_order_gain} | "
            #       f"amm_order_cost: {amm_order_cost} | "
            #       f"reward: {rew}")
            # print(self.amm)
        else:
            success, info = True, {}
            rew = 0.

        # print(f"portfolio A : {self.amm.portfolio['A']} | "
        #       f"portfolio B : {self.amm.portfolio['B']} | "
        #       f"minimum shares : {0.2 * self.shares} | "
        #       f"terminated : {min(self.amm.portfolio['A'], self.amm.portfolio['B']) < 0.2 * self.shares}")
        # print(f"action: {actions} | reward: {rew}")

        # print(self.amm)
        done = False
        if min(self.amm.reserve_a, self.amm.reserve_b) < 0.2 * self.shares:
            done = True
            self.reset()
            # print("terminated, start to reset")
            # print(self.amm)

        self.cum_pnl += rew
        self.market.next()

        next_obs = self.get_obs()

        # Debug prints
        # print(f"Reward: {rew}, Cumulative PnL: {self.cum_pnl}")
        # print(f"Next observation: {next_obs}, Done: {done}")

        return next_obs, rew, done, False, {}

    def reset(self):
        self.cum_pnl = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}

    def get_obs(self) -> np.array:
        cur_market_price = self.market.current_price
        cur_amm_price = self.amm.get_price()
        return np.array([cur_market_price, cur_amm_price, self.amm.reserve_a/10000, self.amm.reserve_b / 10000], dtype=np.float32)

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    market = GBMPriceSimulator()
    fee1 = PercentFee(0.0)
    # fee2 = TriangleFee(0.2, -1)
    # amm = AMM()
    amm = SimpleFeeAMM(fee_structure=fee1)
    env = ArbitrageEnv(market, amm)

    # Reset the environment
    obs = env.reset()
    print(f"Initial observation: {obs}")
    step = 0
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
