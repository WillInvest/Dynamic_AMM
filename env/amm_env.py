import os
from .market import MarketSimulator
from .new_amm import AMM
from typing import Tuple
import numpy as np
from gymnasium import spaces, Env
from stable_baselines3 import PPO
 
class DynamicAMM(Env):
    def __init__(self,
                 market: MarketSimulator,
                 amm: AMM,
                 trader_dir
                ) -> None:
        super().__init__()
        
        self.traders = {}
        for mc in np.arange(0.02, 0.22, 0.02):
            mc = round(mc, 2)
            model_path = os.path.join(trader_dir, f'market_competition_level_{mc:.2f}', 'best_model.zip')
            self.traders[mc] = PPO.load(model_path)
            print(f"Loaded trader with competition level of {mc:.2f}")
        assert len(self.traders) > 0, "No traders loaded"
        self.num_traders = len(self.traders)

        self.amm = amm
        self.market = market
        self.step_count = 0
        self.fee_rate = None
        self.done = False
        self.max_steps = 500
        self.fee_scaler = 0.1
        self.current_action = None
        self.urgent_levels = []  # List to store urgent levels
        self.swap_rates = [np.round(rate, 2) for rate in np.arange(-1.0, 1.1, 0.1) if np.round(rate, 2) != 0]
        self.total_pnl = {mc: 0.0 for mc in self.traders.keys()}
        self.total_fee = {mc: 0.0 for mc in self.traders.keys()}

        # observation space
        low = np.zeros(4)
        high = np.inf * np.ones(4)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # action space
        self.action_space = spaces.Discrete(self.amm.num_slices)

        # self.action_space = spaces.Box(low=np.array([0.]),
        #                                high=np.array([1.]), dtype=np.float32)
        
    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        """
        urgent level determine whether agent place order or not.
        market competence determine how much percent of arbitrage oppotunity will be taken by other traders in the market
        """
        # Initialize step pnl and fee
        reward = 0
        swap_rates = {mc: 0 for mc in self.traders.keys()}
        self.current_action = action
        self.fee_rate = round(self.amm.fee_rates[self.current_action], 4)
        self.amm.fee = self.fee_rate
        traders_to_process = list(self.traders.keys())
        trader_obs = self.get_trader_obs()
        trader_actions = []
            
        for mc in traders_to_process:
            trader = self.traders[mc]
            action, _states = trader.predict(trader_obs)
            swap_rate = self.swap_rates[action[0]] * 0.1
            urgent_level = self.amm.fee_rates[action[1]]
            trader_actions.append((urgent_level, swap_rate, mc))
            
        # Sort by urgent level and get the highest urgency level trader
        trader_actions.sort(reverse=True, key=lambda x: x[0])
        for action in trader_actions:
            urgent_level, swap_rate, mc = action
            if urgent_level >= self.amm.fee:
                # TODO: create a fake AMM to test whether the swap will generate positive PnL
                swap_rates[mc] = swap_rate
                # check profit availability by simulating the swap; if positive, there is remaining arbitrage, then execute the swap
                # simu_info = self.amm.simu_swap(swap_rate)
                # simu_pnl, simu_fees = self.calculate_pnl(simu_info, swap_rate)
                # if simu_pnl > 0:
                #     info = self.amm.swap(swap_rate)
                #     pnl, fees = self.calculate_pnl(info, swap_rate)
                #     self.total_pnl[mc] += pnl
                #     self.total_fee[mc] += fees
                #     reward += fees
                info = self.amm.swap(swap_rate)
                pnl, fees = self.calculate_pnl(info, swap_rate)
                if pnl > 0:
                    self.total_pnl[mc] += pnl
                    self.total_fee[mc] += fees
                    reward += fees
            else:
                # If the highest urgency level is not higher than the fee rate, stop processing
                break
            
        infos = {
            "cumulative_fee": sum(self.total_fee.values()),
            "cumulative_pnl": sum(self.total_pnl.values()),
            "total_pnl": self.total_pnl,
            "total_fee": self.total_fee,
            "swap_rates": swap_rates
            }

        # increase the step count
        self.step_count += 1
        if self.step_count == self.market.steps or min(self.amm.reserve_a, self.amm.reserve_b) < self.amm.initial_shares * 0.2:
            self.done = True
        # Advance market to the next state
        self.market.next()
        next_obs = self.get_obs()
        
        return next_obs, reward, self.done, False, infos
    
    
    def calculate_pnl(self, info, swap_rate):
        if swap_rate < 0:
            asset_in, asset_out = 'A', 'B'
        else:
            asset_in, asset_out = 'B', 'A'
        asset_delta = info['asset_delta']
        fee = info['fee']
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * self.market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * self.market.get_bid_price(asset_out)
        pnl = (market_gain - amm_cost) / self.market.initial_price if swap_rate != 0 else 0  
        fees = (fee['A'] + fee['B']) / self.market.initial_price
        return pnl, fees

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.done = False
        self.step_count = 0
        self.total_pnl = {mc: 0.0 for mc in self.traders.keys()}
        self.total_fee = {mc: 0.0 for mc in self.traders.keys()}
        self.amm.reset()
        self.market.reset()
        obs = self.get_obs()
        return obs, {}
    
    def get_obs(self) -> np.array:
        obs = np.array([
                  self.market.get_ask_price('A') / self.market.get_bid_price('B'),
                  self.market.get_bid_price('A') / self.market.get_ask_price('B'),
                  self.amm.reserve_b / self.amm.initial_shares,
                  self.amm.reserve_a / self.amm.initial_shares
                  ], dtype=np.float32)
        return obs

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
 