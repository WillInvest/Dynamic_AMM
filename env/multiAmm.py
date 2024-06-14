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
        # observation space = [market price, amm price, gas price]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.], dtype=np.float32),
                                            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32))
        # action space = [swap percentage, gas fee]
        self.action_space = spaces.Box(low=np.array([-1., 0.]),
                                       high=np.array([1., 1.]), dtype=np.float32)
        self.done = False
        self.rule_based = rule_based
        
        self.agent1 = TD3.load(model_path)
        print(f"agent loaded from {model_path}")

    def step(self, actions1: np.array) -> Tuple[np.array, float, float, bool, dict]:  
        # actions1, actions2 = actions1
        # Scale action to avoid depletion too quickly
        swap_rate1 = actions1[0] * 0.2
        tip_rate1 = actions1[1] * 0.1
        obs = self.get_obs()
        actions2, _state = self.agent1.predict(obs) 
        swap_rate2 = actions2[0] * 0.2
        if self.rule_based:
            tip_rate2 = 0 + abs((obs[1] - obs[0]) / obs[1]) * 5 # 5 is the predetermined maximum tip rate
        else:
            tip_rate2 = actions2[1]
            
        # rescale the tip rates
        tip_rate1 *= 0.01
        tip_rate2 *= 0.01
        
        def execute_trade(swap_rate, tip_rate, asset_in, asset_out):
            post_fee_rate = self.fee_rate + tip_rate # update amm with new fee rate due to tip rate
            self.amm.fee = post_fee_rate
            info = self.amm.swap(swap_rate)
            self.amm.fee = self.fee_rate   # set back to the original rate
            asset_delta = info['asset_delta']
            fee = info['fee']
            amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
            market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
            reward = (market_gain - amm_cost) / self.market.initial_price 
            return reward, fee

        def determine_assets(swap_rate):
            if swap_rate > 0:
                return 'B', 'A'
            else:
                return 'A', 'B'

        def process_trades(swap_rate1, tip1, swap_rate2, tip2):
            rew1 = rew2 = 0
            fee1 = fee2 = {'A': 0, 'B': 0}  # Initialize fees to avoid reference before assignment

            if tip1 >= tip2:
                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1 = execute_trade(swap_rate1, tip1, asset_in1, asset_out1)

                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2 = execute_trade(swap_rate2, tip2, asset_in2, asset_out2)
            else:
                asset_in2, asset_out2 = determine_assets(swap_rate2)
                rew2, fee2 = execute_trade(swap_rate2, tip2, asset_in2, asset_out2)

                asset_in1, asset_out1 = determine_assets(swap_rate1)
                rew1, fee1 = execute_trade(swap_rate1, tip1, asset_in1, asset_out1)

            return rew1, rew2, fee1, fee2

        # Example usage:
        rew1, rew2, fee1, fee2 = process_trades(swap_rate1, tip_rate1, swap_rate2, tip_rate2)


        # Update cumulative fee
        self.cumulative_fee += (fee1['A'] + fee1['B'] + fee2['A'] + fee2['B'])
        '''
        Apply punishment if rewards are negative
        push both agents to make positive reward
        instead of sacrifying one and make another one more profitable
        '''
        punishment_factor = 2 
        main_reward_factor = 2
        if rew1 < 0:
            rew1 *= punishment_factor
        else:
            rew1 *= main_reward_factor
        if rew2 < 0:
            rew2 *= punishment_factor
        # Update the total rewards
        self.total_rewards += (rew1 + rew2)        
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

        # Advance market and gas price to the next state
        self.market.next()
        next_obs = self.get_obs()

        return next_obs, rew1, self.done, False, {}

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
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
 
if __name__ == "__main__":
    from exp.amm_ddpg import AgentDDPG 

    model_path = '/home/shiftpub/AMM-Python/stable_baseline/single_agent/models/TD3/2024-06-10_10-08-52/agent_seed_0/fee_0.01/sigma_0.2/TD3_910000'
    market = MarketSimulator(start_price=1, deterministic=False)
    amm = AMM(initial_a=8000, initial_b=10000, fee=0.02)  # Set your fee rate
    env = MultiAgentAmm(market, amm, model_path=model_path)
    
    obs, _ = env.reset()
    
    action = np.array([0.1, 0.1])
    
    obs, rew, done, truncated, info = env.step(action)

    
    # for _ in range(100):
    
    #     obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device='cuda:0').unsqueeze(0)
    #     action1 = agent.act1.get_action(obs_tensor).detach().cpu().numpy()[0]
    #     action2 = agent.act2.get_action(obs_tensor).detach().cpu().numpy()[0]
    #     print(f"action1: {action1} | action2: {action2}")

    #     obs, rew1, rew2, done, truncated, info = env.step(action1, action2)
        
    #     print(f"obs: {obs} | rew1: {rew1} | rew2: {rew2} | done: {done}")
        
    #     if done:
    #         obs, _ = env.reset()

        
        
        