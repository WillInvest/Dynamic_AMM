import os
from .market import MarketSimulator
from .new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import PPO

EPSILON = 1e-5
 
class DynamicAMM(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 trader_dir
                ) -> None:
        super().__init__()
        self.amm = amm
        self.market = market
        self.cumulative_fee = 0
        self.step_count = 0
        self.fee_rate = None
        self.done = False
        self.max_steps = 500
        self.reward_scaler = 0.1
        self.lwf = 0.0001
        self.upf = 0.005
        self.base_fee_rate = 0.003
        self.fee_range = 0.002
        self.action = None

        
        self.traders = {}
        for mc in np.arange(0.01, 0.10, 0.01):
            model_path = os.path.join(trader_dir, f'market_competence_{mc:.2f}', 'rl_model_9920000_steps.zip')
            if os.path.exists(model_path):
                self.traders[mc] = PPO.load(model_path)
                print(f"Loaded model for market competence {mc:.2f}")
                
        assert len(self.traders) > 0, "No traders loaded"
        
        self.num_traders = len(self.traders)
        
        # observation space
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0.]),
                                        high=np.array([np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
        
        # action space
        self.action_space = spaces.Box(low=np.array([0.]),
                                       high=np.array([1.]), dtype=np.float32)
        
    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """
        urgent level determine whether agent place order or not.
        market competence determine how much percent of arbitrage oppotunity will be taken by other traders in the market
        """
        # get the fee rate
        self.action = action[0]
        self.fee_rate = self.action * (self.upf - self.lwf) + self.lwf
        # get the trader observation
        trader_obs = self.get_trader_obs()
        actions = []
        for competence, trader in self.traders.items():
            action, _states = trader.predict(trader_obs)
            swap_rate, urgent_level = action
            swap_rate *= 0.2
            if urgent_level >= self.fee_rate:  # Filter actions based on fee rate
                actions.append((urgent_level, swap_rate))
        
        # Sort by urgent level
        actions.sort(reverse=True, key=lambda x: x[0])
        
        # Execute trades in order of urgent level
        self.cumulative_fee = 0
        for urgent_level, swap_rate in actions:
            info = self.amm.swap(swap_rate)
            fee = info['fee']
            self.cumulative_fee += (fee['A'] + fee['B']) * self.reward_scaler
        
        # increase the step count
        self.step_count += 1
        
        if self.market.index == self.market.steps or min(self.amm.reserve_a, self.amm.reserve_b) < self.amm.initial_shares * 0.2:
            self.done = True

        # Advance market to the next state
        self.market.next()
        next_obs = self.get_obs()

        return next_obs, self.cumulative_fee, self.done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.cumulative_fee = 0
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}
    
    
    def get_obs(self) -> np.array:
        return np.array([
            self.market.get_ask_price('A') / self.market.get_bid_price('B'),
            self.market.get_bid_price('A') / self.market.get_ask_price('B'),
            self.amm.reserve_b / self.amm.initial_shares,
            self.amm.reserve_a / self.amm.initial_shares
            ], dtype=np.float32)
 
    # def get_obs(self) -> np.array:
    #     """
    #     return states:
    #     1) market ask price
    #     2) market bid price
    #     3) amm mid price
    #     4) trader states (flattened)
    #     """
    #     urgent_levels = []
    #     swap_rates = []
    #     trader_obs = self.get_trader_obs(type='base')
    #     for competence, trader in self.traders.items():
    #         action, _states = trader.predict(trader_obs)
    #         swap_rate, urgent_level = action
    #         urgent_levels.append(urgent_level)
    #         swap_rates.append(swap_rate)
        
    #     # Flatten the trader state lists
    #     trader_state_flattened = np.array(urgent_levels + swap_rates, dtype=np.float32)

    #     # Create the main observation array
    #     main_obs = np.array([
    #         self.market.get_ask_price('A') / self.market.get_bid_price('B'),
    #         self.market.get_bid_price('A') / self.market.get_ask_price('B'),
    #         self.amm.reserve_b / self.amm.reserve_a
    #     ], dtype=np.float32)

    #     # Concatenate the main observation with the flattened trader state
    #     combined_obs = np.concatenate((main_obs, trader_state_flattened))

    #     return combined_obs


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
            (self.amm.reserve_b / self.amm.reserve_a) * (1+self.fee_rate),
            (self.amm.reserve_b / self.amm.reserve_a) / (1+self.fee_rate),
            self.fee_rate
            ], dtype=np.float32)
 
    def render(self, mode='human'):
        pass
 
if __name__ == "__main__":
    MODEL_DIR = '/home/shiftpub/AMM-Python/stable_baseline/multiple_agent/models'
    market = MarketSimulator(start_price=1, deterministic=True)
    amm = AMM(initial_a=8000, initial_b=10000, fee=0.02)  # Set your fee rate
    env = DynamicAMM(market, amm, trader_dir=MODEL_DIR)
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    action = env.action_space.sample()
    next_obs, reward, done, _, _ = env.step(action)
    print(f"total_fee: {env.cumulative_fee}")
    
    
    

        
        
        