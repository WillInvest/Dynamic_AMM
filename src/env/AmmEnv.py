import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market import GBMPriceSimulator
from new_amm import AMM
# from amm.amm import AMM, SimpleFeeAMM
# from amm.fee import PercentFee
# from amm.utils import parse_input
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
        self.cum_pnl = 0.
        self.USING_USD = USD
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32))
        self.action_space = spaces.Box(low=np.array([-1 + EPSILON], dtype=np.float32),
                                       high=np.array([1 - EPSILON], dtype=np.float32), dtype=np.float32)

    # action: [trade_size_fraction, trade_decision (take opportunity when >0.5)]

    def step(self, actions: np.array) -> Tuple[
        np.array, float, bool, bool, dict]:  # next_obs, rew, done, truncated, info
        actions = actions * 0.1
        info = self.amm.swap(actions)
        # calculate the reward
        asset_delta = info['asset_delta']
        # print(asset_delta)
        fee = info['fee']
        if self.USING_USD:
            if actions > 0:
                asset_in, asset_out = 'B', 'A'
            else:
                asset_in, asset_out = 'A', 'B'
        # print(f"fee: {fee} | asset_delta : {asset_delta}")
        # print(f"market_epsilon: {self.market.epsilon}")
  
            amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
            market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
            rew = (market_gain - amm_cost) / self.market.initial_price
        else:
            amm_order_cost = asset_delta['B'] + fee['B']  # unit is always in B
            market_order_gain = (asset_delta['A'] + fee['A']) * (
                self.market.get_bid_price('B') if asset_delta['A'] < 0 else self.market.get_ask_price('B'))
            # print(f"market_bid: {self.market.get_bid_price()} | market_ask: {self.market.get_ask_price()}")
            # print(f"amm_bid: {self.amm.get_price()/(1+0.02)} | amm_ask: {self.amm.get_price()*(1+0.02)}")
            rew = - (market_order_gain + amm_order_cost)

        #
        # print(f"asset_delta['A'] : {asset_delta['A']}, fee['FA']: {fee['A']} | "
        #         f"asset_delta['B']: {asset_delta['B']} | "
        #         f"asset_A : {self.amm.reserve_a} | asset_B : {self.amm.reserve_b} | ")
        # print(f"market_order_gain: {market_gain} | "
        #         f"amm_order_cost: {amm_cost} | "
        #         f"reward: {rew}")
        # print(self.amm)
   

        # print(f"portfolio A : {self.amm.portfolio['A']} | "
        #       f"portfolio B : {self.amm.portfolio['B']} | "
        #       f"minimum shares : {0.2 * self.shares} | "
        #       f"terminated : {min(self.amm.portfolio['A'], self.amm.portfolio['B']) < 0.2 * self.shares}")
        # print(f"action: {actions} | reward: {rew}")

        # print(self.amm)
        done = False
        # print(f"shock_index: {self.market.shock_index}")
        condition = self.market.shock_index % 499 == 0 and self.market.shock_index >= 50
        # print(f"cpndition: {condition} | {self.market.shock_index}")
        if min(self.amm.reserve_a, self.amm.reserve_b) < 0.2 * self.shares or condition:
            done = True
            self.reset()
            # print("terminated, start to reset")
            # print(self.amm)

        self.cum_pnl += rew
        self.market.next()
        # print(f"market_price: {self.market.current_price}")

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
        return obs

    def get_obs(self) -> np.array:
        if self.USING_USD:
            if isinstance(self.amm.reserve_a, np.ndarray) and self.amm.reserve_a.size == 1:
                self.amm.reserve_a = self.amm.reserve_a.item()
            if isinstance(self.amm.reserve_b, np.ndarray) and self.amm.reserve_b.size == 1:
                self.amm.reserve_b = self.amm.reserve_b.item()    
            return np.array([
                self.amm.reserve_a/self.shares, 
                self.amm.reserve_b/self.shares, 
                self.market.AP/self.market.initial_price, 
                self.market.BP/self.market.initial_price
                             ], dtype=np.float32)
        else:
            cur_market_price = self.market.BP
            cur_amm_price = self.amm.get_price()
            if isinstance(cur_market_price, np.ndarray) and cur_market_price.size == 1:
                cur_market_price = cur_market_price.item()

            if isinstance(cur_amm_price, np.ndarray) and cur_amm_price.size == 1:
                cur_amm_price = cur_amm_price.item()
            return np.array([cur_market_price,cur_amm_price], dtype=np.float32)


        # print(f"market_shape: {cur_market_price.shape}")
        # print(f"market type: {type(cur_market_price)}")
        # print(f"amm_shape: {cur_amm_price.shape}")
        # print(f"amm type: {type(cur_amm_price)}")
            
            

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    market = GBMPriceSimulator()

    amm = AMM(initial_a=10000, initial_b=10000, fee=0.0)
    env = ArbitrageEnv(market, amm)

    # Reset the environment
    next_obs = env.reset()
    print(f"Initial observation: {next_obs}")
    step = 0
    # Perform a few steps with sample actions
    for step in range(10):
        # Generate a random action
        trade_size_fraction = np.random.uniform(-0.1, 0.1)  # Random number between -1 and 1
        trade_decision = np.random.uniform(0, 1)  # Random number between 0 and 1
        # action = np.array([trade_size_fraction, trade_decision], dtype=np.float32)
        # action[1] = 1
        # s2string = input("Input string (i.e. A B 1): ")
        # if s2string == 'r':
        #     env.reset()
        #     continue  # reset
        action = trade_size_fraction
        fee_rate = 0
        print(f"\n--------------------"
              f"\nStep {step + 1}:")
        print(f"action: {action}")
        print(f"A market price: {market.get_ask_price("A")} | {market.get_bid_price("A")}")
        print(f"B market price: {market.get_ask_price("B")} | {market.get_bid_price("B")}")
        print(f"amm_ask_ratio: {(next_obs[1]/next_obs[0]) * (1+fee_rate)} | amm_bid_ratio: {(next_obs[1]/next_obs[0]) / (1+fee_rate)}")
        print(f"market_ask_ratio: {market.get_ask_price("A") / market.get_bid_price("B")} | market_bid_ratio: {market.get_bid_price("A") / market.get_ask_price("B")}")
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
