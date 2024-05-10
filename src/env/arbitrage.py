from market import GBMPriceSimulator
from amm.amm import AMM
from typing import Tuple
import numpy as np
from gym.spaces import Discrete, Box

EPSILON = 1e-5

class ArbitrageEnv:
    def __init__(self, 
                 market: GBMPriceSimulator,
                 amm: AMM 
                 ) -> None:
        self.amm = amm
        self.market = market
        self.cum_pnl = 0.
        self.observation_space = Box(low=np.array([0., 0., 0., 0. ]), high=np.array([np.inf, np.inf, np.inf, np.inf])) # obs: [market price, amm price, amm inv A, amm inv B]
        self.action_space = Box(low=np.array[-1+EPSILON, 0], high=np.array([1-EPSILON, 1])) # action: [trade_size_fraction, trade_decision (take opportunity when >0.5)]
        
    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]: # next_obs, rew, done, truncated, info
        trade_size_fraction, trade_prob = action
        if trade_prob > 0.5:
            if trade_size_fraction > 0:
                asset_in, asset_out = 'B', 'A'
            else:
                asset_in, asset_out = 'A', 'B'
            size = abs(trade_size_fraction)*self.amm.portfolio[asset_out]
            success, info = self.amm.trade_swap(asset_in, asset_out, -size) # take out -size shares of outing asset
            
            # calculate the reward
            asset_delta = info['asset_delta']
            fee = info['fee']
            amm_order_cost = asset_delta['B'] + fee['B'] # unit is always in B 
            market_order_gain = (asset_delta['A'] + fee['A'])* (self.market.get_bid_price() if asset_delta['A'] < 0 else self.market.get_ask_price()) # market unit xxx B/A
            rew = - (market_order_gain + amm_order_cost)
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
        
    
    def get_obs(self) -> np.array:
        cur_market_price = self.market.current_price
        tmp = self.amm.portfolio
        cur_amm_price = tmp['A']/tmp['B']
        
        return np.array(cur_market_price, cur_amm_price, tmp['A'], tmp['B'])
    

        