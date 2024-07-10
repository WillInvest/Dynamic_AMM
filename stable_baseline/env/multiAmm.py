import sys
import os
 
from .market import MarketSimulator
from .new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import TD3
import torch
import math
import random
EPSILON = 1e-5
 
class MultiAgentAmm(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 shares=10000,
                 model_path = None,
                 rule_based = True,
                 risk_aversion = 0.2
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
        self.risk_aversion = risk_aversion
        # observation space = [market price, amm price, gas price]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32))
        # action space = [swap percentage, gas fee]
        self.action_space = spaces.Box(low=np.array([-1., 0.]),
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
        # print(f"obs: {obs}")
        # print(f"amm_ask: {obs[1] * (1+self.fee_rate)} | amm_bid: {obs[1] / (1+self.fee_rate)} | market_ask: {self.market.get_ask_price('A') / self.market.get_bid_price('B')} | market_bid: {self.market.get_bid_price('A') / self.market.get_ask_price('B')}")

        if self.rule_based:
            amm_ask = obs[1] * (1 + self.fee_rate)
            amm_bid = obs[1] / (1 + self.fee_rate)
            market_ask = self.market.get_ask_price('A') / self.market.get_bid_price('B')
            market_bid = self.market.get_bid_price('A') / self.market.get_ask_price('B')
            if amm_ask < market_bid:
                swap_rate2 = 1 - math.sqrt(self.amm.reserve_a * self.amm.reserve_b / (market_bid/(1+self.amm.fee))) / self.amm.reserve_a
                delta_a = self.amm.reserve_a - math.sqrt(self.amm.reserve_a * self.amm.reserve_b / (market_bid/(1+self.amm.fee)))
                delta_b = self.amm.reserve_a * self.amm.reserve_b / (self.amm.reserve_a - delta_a) - self.amm.reserve_b
                potential_market_gain = delta_a * self.market.get_bid_price('A') 
                potential_amm_cost = delta_b * self.market.get_ask_price('B') * (1+self.fee_rate)
                arbitrage_profit = potential_market_gain - potential_amm_cost
                gas_fee2 = arbitrage_profit * self.risk_aversion # 0.2 is the risk aversion factor
                # print(f"amm_ask<market_bid: {amm_ask} < {market_bid} | swap_rate2: {swap_rate2} | delta_a: {delta_a} | delta_b: {delta_b} | gas_fee2: {gas_fee2} | arbitrage_profit: {arbitrage_profit}")
                # print(f"market_a: {self.market.get_bid_price('A')} | market_b: {self.market.get_ask_price('B')}")
            elif amm_bid > market_ask:
                swap_rate2 = math.sqrt((self.amm.reserve_a*self.amm.reserve_b*market_ask*(1+self.amm.fee)))/self.amm.reserve_b - 1
                # swap_rate2 = - (1 - math.sqrt(self.amm.reserve_a * self.amm.reserve_b * market_ask / (1-self.amm.fee)) / self.amm.reserve_b)
                delta_b = self.amm.reserve_b - math.sqrt(self.amm.reserve_a * self.amm.reserve_b * market_ask*(1+self.amm.fee))
                delta_a = self.amm.reserve_a * self.amm.reserve_b / (self.amm.reserve_b - delta_b) - self.amm.reserve_a
                potential_market_gain = delta_b * self.market.get_bid_price('B')
                potential_amm_cost = delta_a * self.market.get_ask_price('A') * (1+self.fee_rate)
                arbitrage_profit = potential_market_gain - potential_amm_cost
                gas_fee2 = arbitrage_profit * self.risk_aversion # 0.2 is the risk aversion factor
                # print(f"amm_bid>market_ask: {amm_bid} > {market_ask} | swap_rate2: {swap_rate2} | gas_fee2: {gas_fee2} | arbitrage_profit: {arbitrage_profit}")
            else:
                # print(f"amm_ask: {amm_ask} | amm_bid: {amm_bid} | market_ask: {market_ask} | market_bid: {market_bid}")
                swap_rate2 = 0
                gas_fee2 = 0
        else:
            actions2, _state = self.agent1.predict(obs) 
            # swap_rate2, gas_fee2 = actions2[0] * 0.2, actions2[1] * 1e-4
            swap_rate2, gas_fee2 = 0, 0
        
        market_ask = self.market.get_ask_price('A') / self.market.get_bid_price('B')
        market_bid = self.market.get_bid_price('A') / self.market.get_ask_price('B')
        amm_mid = obs[1]
        if amm_mid < market_bid:
            delta_a = self.amm.reserve_a - math.sqrt(self.amm.reserve_a * self.amm.reserve_b / market_bid)
            delta_b = self.amm.reserve_a * self.amm.reserve_b / (self.amm.reserve_a - delta_a) - self.amm.reserve_b
            potential_market_gain = delta_a * self.market.get_bid_price('A') 
            potential_amm_cost = delta_b * self.market.get_ask_price('B')
            fake_arbitrage_profit = potential_market_gain - potential_amm_cost
            # print(f"amm_mid<market_mid, delta_a: {delta_a} | delta_b: {delta_b} | market_a: {self.market.get_bid_price('A')} | market_b: {self.market.get_ask_price('B')} | potential_market_gain: {potential_market_gain} | potential_amm_cost: {potential_amm_cost}")
        elif amm_mid > market_ask:
            delta_b = self.amm.reserve_b - math.sqrt(self.amm.reserve_a * self.amm.reserve_b * market_ask)
            delta_a = self.amm.reserve_a * self.amm.reserve_b / (self.amm.reserve_b - delta_b) - self.amm.reserve_a
            potential_market_gain = delta_b * self.market.get_bid_price('B')
            potential_amm_cost = delta_a * self.market.get_ask_price('A')
            fake_arbitrage_profit = potential_market_gain - potential_amm_cost
            # print(f"amm_mid>market_mid, delta_a: {delta_a} | delta_b: {delta_b} | market_a: {self.market.get_ask_price('A')} | market_b: {self.market.get_bid_price('B')} | potential_market_gain: {potential_market_gain} | potential_amm_cost: {potential_amm_cost}")

        else:
            fake_arbitrage_profit = 0
            delta_a = delta_b = 0
        
        swap_rate1, gas_fee1 = actions1[0] * 0.2, max(abs(actions1[1] * fake_arbitrage_profit), 1e-4)
        
        def execute_trade(swap_rate, gas_fee, asset_in, asset_out):
            info = self.amm.swap(swap_rate)
            self.amm.fee = self.fee_rate   # set back to the original rate
            asset_delta = info['asset_delta']
            fee = info['fee']
            amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
            market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
            reward = (market_gain - amm_cost - gas_fee)/self.amm.initial_a
            return reward, fee

        def determine_assets(swap_rate):
            if swap_rate > 0:
                return 'B', 'A'
            else:
                return 'A', 'B'

        def process_trades(swap_rate1, gas1, swap_rate2, gas2):
            rew1 = rew2 = 0
            fee1 = fee2 = {'A': 0, 'B': 0}
            if gas1 >= gas2:
                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1 = execute_trade(swap_rate1, gas1, asset_in1, asset_out1)
                if gas1 == gas2:
                    print("Gas fees are equal")

                # asset_in2, asset_out2 = determine_assets(swap_rate2)
                # rew2, fee2 = execute_trade(swap_rate2, gas2, asset_in2, asset_out2)
            else:
                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2 = execute_trade(swap_rate2, gas2, asset_in2, asset_out2)

                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1 = execute_trade(swap_rate1, gas1, asset_in1, asset_out1)

            return rew1, rew2, fee1, fee2

        # process trades
        rew1, rew2, fee1, fee2 = process_trades(swap_rate1, gas_fee1, swap_rate2, gas_fee2)


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
        # modified_rew1 = 2 * rew1 if rew1<0 else rew1
        # modified_rew2 = 2 * rew2 if rew2<0 else rew2
        # total_rewards = 2 * modified_rew1 + modified_rew2
        self.total_rewards += rew1 + rew2
        self.total_gas += gas_fee1 + gas_fee2

        # Advance market and gas price to the next state
        self.market.next()
        next_obs = self.get_obs()

        return next_obs, rew1, self.done, False, {'total_rewards': (self.total_rewards),
                                                           'reward1': rew1,
                                                           'reward2': rew2,
                                                           'total_gas': (gas_fee1 + gas_fee2)}

    
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
        """
        return 6 states
        1) market mid price ratio
        2) amm mid price ratio
        3) amm reserve normalized token A
        4) amm reserve normalized token B
        5) market normalized price for token A
        6) market normalized price for token B
        """
        return np.array([self.market.AP / self.market.BP,
                         self.amm.reserve_b / self.amm.reserve_a,
                         self.amm.reserve_a/self.amm.initial_a,
                         self.amm.reserve_b/self.amm.initial_b,
                         self.market.AP/self.market.initial_price,
                         self.market.BP/self.market.initial_price], dtype=np.float32)
 
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

        
        
        