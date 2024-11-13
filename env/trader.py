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
        self.reset()
    
    def reset(self):
        self.pnl = 0
        self.total_number_trade = 0
        self.total_fee = 0
        self.amm.reset()
        self.oracle.reset()
        
    def swap(self):
        mkt_ask, mkt_bid = self.oracle.get_price()   
        amm_ask, amm_bid = self.amm.get_price()
        k = self.amm.ls * self.amm.lr
        if self.amm.distribute:
            if amm_ask < mkt_bid:
                x_r = np.sqrt(k/(mkt_bid * (1-self.amm.f))) - self.amm.lr
                swap_info = self.amm.swap(x_r)
            elif amm_bid > mkt_ask:
                x_r = (np.sqrt(k * (1-self.amm.f) / mkt_ask) - self.amm.lr) / (1-self.amm.f)
                swap_info = self.amm.swap(x_r)
            else:
                swap_info = self.amm.swap(0)
        else:
            if amm_ask < mkt_bid:
                a = 1 - self.amm.f
                b = (2-self.amm.f) * self.amm.ls
                c = self.amm.ls**2 - self.amm.ls*self.amm.lr*(1-self.amm.f)*mkt_bid
                x_s = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                x_r = -self.amm.lr * (1-self.amm.f) * x_s / (self.amm.ls + (1-self.amm.f) * x_s)
                swap_info = self.amm.swap(x_r)
            elif amm_bid > mkt_ask:
                a = 1 - self.amm.f
                b = (2-self.amm.f) * self.amm.lr
                c = self.amm.lr**2 - self.amm.ls*self.amm.lr*(1-self.amm.f)/mkt_ask
                x_r = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                swap_info = self.amm.swap(x_r)
            else:
                swap_info = self.amm.swap(0)
                
        swap_info.update(self.update_info(swap_info, prev_amm_ask=amm_ask, prev_amm_bid=amm_bid))
        
        return swap_info
        
    
    def update_info(self, swap_info, prev_amm_ask, prev_amm_bid):
        # Determine the asset in/out and tokens based on 'xr' value
        asset_in, asset_out = (swap_info['xr'], swap_info['xs']) if swap_info['xr'] > 0 else (swap_info['xs'], swap_info['xr'])
        token_in, token_out = ('r', 's') if swap_info['xr'] > 0 else ('s', 'r')

        # Calculate costs and gains
        ask_price_in = self.oracle.get_ask_price(token_in)
        bid_price_out = self.oracle.get_bid_price(token_out)
        fee_cost = swap_info['token_fee'] * ask_price_in
        amm_cost = asset_in * ask_price_in
        mkt_gain = -asset_out * bid_price_out

        # Calculate PnL and update totals
        pnl = mkt_gain - amm_cost - fee_cost
        self.pnl += pnl
        
        
        if swap_info['xr'] != 0:
            self.total_number_trade += 1
            self.total_fee += fee_cost
        initial_pool_value = self.amm.initial_lr * self.oracle.get_mid_price('r') + self.amm.initial_ls * self.oracle.get_mid_price('s')
        current_pool_value = self.amm.lr * self.oracle.get_mid_price('r') + self.amm.ls * self.oracle.get_mid_price('s')
        impermanent_loss = current_pool_value - initial_pool_value
        net_profit = self.total_fee + impermanent_loss
            
        pnl_info = {
            'pnl': pnl,
            'fee_dollar_value': fee_cost,
            'total_fee_dollar_value': self.total_fee,
            'mkt_gain': mkt_gain,
            'amm_cost': amm_cost,
            'initial_pool_value': initial_pool_value,
            'current_pool_value': current_pool_value,
            'impermanent_loss': impermanent_loss,
            'net_profit': net_profit,
            'total_number_trade': self.total_number_trade,
            'token_in': token_in,
            'token_out': token_out,
            'asset_in': asset_in,
            'asset_out': asset_out,
            'mkt_ask': self.oracle.get_price()[0],
            'mkt_bid': self.oracle.get_price()[1],
            'prev_amm_ask': prev_amm_ask,
            'prev_amm_bid': prev_amm_bid,
            'amm_ask': self.amm.get_price()[0],
            'amm_bid': self.amm.get_price()[1],
            'spread': self.oracle.spread,
            'fee_rate': self.amm.f,
            'fee_pool': self.amm.distribute,
            'mid_r': self.oracle.get_mid_price('r'),
            'mid_s': self.oracle.get_mid_price('s')
        }

        return pnl_info

            
        
