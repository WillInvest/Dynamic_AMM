import numpy as np
from env.oracle import OracleSimulator

class Arbitrager:
    def __init__(self, oracle: OracleSimulator):
        self.oracle = oracle
        self.amm = self.oracle.amm
        self.reset()
    
    def reset(self):
        self.pnl = 0
        self.total_number_trade = 0
        self.total_fee = 0
        self.oracle.reset()
    
    def step(self):
        self.oracle.next()
        return self.swap()
    
    def _calculate_swap_amount(self, amm_ask, amm_bid, mkt_ask, mkt_bid):
        k = self.amm.ls * self.amm.lr
        f = self.amm.f
        
        if self.amm.fee_distribute:
            if amm_ask < mkt_bid:
                return np.sqrt(k/(mkt_bid * (1-f))) - self.amm.lr
            elif amm_bid > mkt_ask:
                return (np.sqrt(k * (1-f) / mkt_ask) - self.amm.lr) / (1-f)
        else:
            if amm_ask < mkt_bid:
                a = 1 - f
                b = (2-f) * self.amm.ls
                c = self.amm.ls**2 - self.amm.ls*self.amm.lr*(1-f)*mkt_bid
                x_s = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                return -self.amm.lr * (1-f) * x_s / (self.amm.ls + (1-f) * x_s)
            elif amm_bid > mkt_ask:
                a = 1 - f
                b = (2-f) * self.amm.lr
                c = self.amm.lr**2 - self.amm.ls*self.amm.lr*(1-f)/mkt_ask
                return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        return 0

    def swap(self):
        amm_ask, amm_bid, mkt_ask, mkt_bid = self.oracle.get_price()
        x_r = self._calculate_swap_amount(amm_ask, amm_bid, mkt_ask, mkt_bid)
        swap_info = self.amm.swap(x_r)
        swap_info.update(self._calculate_trade_metrics(swap_info, amm_ask, amm_bid))
        return swap_info

    def _calculate_trade_metrics(self, swap_info, prev_amm_ask, prev_amm_bid):
        amm_ask, amm_bid, mkt_ask, mkt_bid = self.oracle.get_price()
        
        is_xr_positive = swap_info['xr'] > 0
        asset_in = swap_info['xr'] if is_xr_positive else swap_info['xs']
        asset_out = swap_info['xs'] if is_xr_positive else swap_info['xr']
        token_in = 'r' if is_xr_positive else 's'
        token_out = 's' if is_xr_positive else 'r'

        ask_price_in = self.oracle.get_token_prices(token_in)['ask']
        bid_price_out = self.oracle.get_token_prices(token_out)['bid']
        
        fee_cost = swap_info['token_fee'][token_in] * ask_price_in
        amm_cost = asset_in * ask_price_in
        mkt_gain = -asset_out * bid_price_out
        pnl = mkt_gain - amm_cost - fee_cost

        self.pnl += pnl
        if swap_info['xr'] != 0:
            self.total_number_trade += 1
            self.total_fee += fee_cost

        initial_pool_value = (self.amm.initial_lr * self.oracle.get_token_prices('r')['mid'] + 
                            self.amm.initial_ls * self.oracle.get_token_prices('s')['mid'])
        current_pool_value = (self.amm.lr * self.oracle.get_token_prices('r')['mid'] + 
                            self.amm.ls * self.oracle.get_token_prices('s')['mid'])
        
        return {
            'pnl': pnl,
            'fee_dollar_value': fee_cost,
            'total_fee_dollar_value': self.total_fee,
            'mkt_gain': mkt_gain,
            'amm_cost': amm_cost,
            'initial_pool_value': initial_pool_value,
            'current_pool_value': current_pool_value,
            'impermanent_loss': current_pool_value - initial_pool_value,
            'net_profit': self.total_fee + (current_pool_value - initial_pool_value),
            'total_number_trade': self.total_number_trade,
            'token_in': token_in,
            'token_out': token_out,
            'asset_in': asset_in,
            'asset_out': asset_out,
            'mkt_ask': mkt_ask,
            'mkt_bid': mkt_bid,
            'prev_amm_ask': prev_amm_ask,
            'prev_amm_bid': prev_amm_bid,
            'amm_ask': amm_ask,
            'amm_bid': amm_bid,
            'spread': self.oracle.spread,
            'fee_rate': self.amm.f,
            'fee_pool': self.amm.fee_distribute,
            'mid_r': self.oracle.get_token_prices('r')['mid'],
            'mid_s': self.oracle.get_token_prices('s')['mid']
        }