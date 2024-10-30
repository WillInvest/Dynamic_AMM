import os
import sys
import numpy as np
# Get the path to the Dynamic_AMM directory
sys.path.append(os.path.expanduser("~/Dynamic_AMM"))

from env.amm import AMM
from env.oracle import OracleSimulator

class Arbitrager:
    def __init__(self, amm: AMM, oracle: OracleSimulator):
        self.amm = amm
        self.oracle = oracle
        self.fee_pool = self.amm.fee_pool
        self.reset()
    
    def reset(self):
        self.pnl = 0
        self.total_number_trade = 0
        self.total_fee = 0
        
    def swap(self):
        mkt_ask, mkt_bid = self.oracle.get_price()   
        amm_ask, amm_bid = self.amm.get_price()
        k = self.amm.ls * self.amm.lr
        if self.amm.fee_pool:
            if amm_ask < mkt_bid:
                a = 1
                b = (self.amm.ls * self.amm.f) / (mkt_bid * (1-self.amm.f)**2) + 2*self.amm.lr
                c = self.amm.lr * (self.amm.lr - self.amm.ls/(mkt_bid * (1-self.amm.f)))
                x_r = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                swap_info = self.amm.swap(x_r)
            elif amm_bid > mkt_ask:
                a = 1
                b = ((2-self.amm.f) * self.amm.lr) / (1 - self.amm.f)
                c = self.amm.lr * (self.amm.lr/(1-self.amm.f) - self.amm.ls/mkt_ask)
                x_r = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                swap_info = self.amm.swap(x_r)
            else:
                swap_info = self.amm.swap(0)
        else:
            if amm_ask < mkt_bid:
                x_r = np.sqrt(k/(mkt_bid * (1-self.amm.f))) - self.amm.lr
                swap_info = self.amm.swap(x_r)
            elif amm_bid > mkt_ask:
                x_r = (np.sqrt(k * (1-self.amm.f) / mkt_ask) - self.amm.lr) / (1-self.amm.f)
                swap_info = self.amm.swap(x_r)
            else:
                swap_info = self.amm.swap(0)
                
        pnl_info = self.calculate_pnl_and_fee(swap_info)
        self.update_swap_info(swap_info, amm_ask, amm_bid, pnl_info)
        
        return swap_info
    

    def update_swap_info(self, swap_info, prev_amm_ask, prev_amm_bid, pnl_info):
        swap_info['mkt_ask'] = self.oracle.get_price()[0]
        swap_info['mkt_bid'] = self.oracle.get_price()[1]
        swap_info['prev_amm_ask'] = prev_amm_ask
        swap_info['prev_amm_bid'] = prev_amm_bid
        swap_info['amm_ask'] = self.amm.get_price()[0]
        swap_info['amm_bid'] = self.amm.get_price()[1]
        swap_info['spread'] = self.oracle.spread
        swap_info['fee_rate'] = self.amm.f
        swap_info['fee_pool'] = self.amm.fee_pool
        swap_info.update(pnl_info)
        
    
    def calculate_pnl_and_fee(self, swap_info):
        # Determine the asset in/out and tokens based on 'xr' value
        asset_in, asset_out = (swap_info['xr'], swap_info['xs']) if swap_info['xr'] > 0 else (swap_info['xs'], swap_info['xr'])
        token_in, token_out = ('r', 's') if swap_info['xr'] > 0 else ('s', 'r')

        # Calculate costs and gains
        ask_price_in = self.oracle.get_ask_price(token_in)
        bid_price_out = self.oracle.get_bid_price(token_out)
        fee_cost = swap_info['fee'] * ask_price_in
        amm_cost = asset_in * ask_price_in
        mkt_gain = -asset_out * bid_price_out

        # Calculate PnL and update totals
        pnl = mkt_gain - amm_cost - fee_cost
        self.pnl += pnl
        
        if swap_info['xr'] != 0:
            self.total_number_trade += 1
            self.total_fee += fee_cost
            
        pnl_info = {
            'pnl': pnl,
            'fee_cost': fee_cost,
            'mkt_gain': mkt_gain,
            'amm_cost': amm_cost
        }

        return pnl_info

            
        
