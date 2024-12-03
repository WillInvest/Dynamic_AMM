import numpy as np
from env.oracle import OracleSimulator

class Arbitrager:
    def __init__(self, oracle: OracleSimulator):
        """
        Simulates an arbitrageur trading between AMM and market oracle prices.

        Parameters:
            oracle (OracleSimulator): Price oracle providing market prices

        Methods:
            reset(): Resets trading stats and oracle state
            step(): Advances oracle one step and executes potential arbitrage
            swap(): Calculates and executes optimal arbitrage trade if profitable

        Private Methods:
            _calculate_swap_amount(): Determines optimal trade size based on price differences
                For fee_distribute=True:
                    - When amm_ask < mkt_bid: Buy from AMM, sell to market
                    - When amm_bid > mkt_ask: Buy from market, sell to AMM
                For fee_distribute=False:
                    - Uses quadratic formula to find optimal amounts
               
            _calculate_trade_metrics(): Computes trade results including:
                - PnL and fees
                - Pool value changes and impermanent loss
                - Market prices and spreads
        """
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
        xr = 0
        if self.amm.fee_source > 0:
            if amm_ask < mkt_bid:
                xr = np.sqrt(k / (mkt_bid * (1-f))) - self.amm.lr
                assert xr < 0, f"xr={xr}, amm_ask={amm_ask}, mkt_bid={mkt_bid}"
            elif amm_bid > mkt_ask:
                xr = (np.sqrt(k * (1-f) / mkt_ask) - self.amm.lr) / (1-f)
                assert xr > 0, f"xr={xr}, amm_bid={amm_bid}, mkt_ask={mkt_ask}"
        else:
            if amm_ask < mkt_bid:
                xr = np.sqrt(k / (mkt_bid * (1-f))) - self.amm.lr
                assert xr < 0, f"xr={xr}, amm_ask={amm_ask}, mkt_bid={mkt_bid}"
            if amm_bid > mkt_ask:
                xr = np.sqrt(k * (1-f) / mkt_ask) - self.amm.lr
                assert xr > 0, f"xr={xr}, amm_bid={amm_bid}, mkt_ask={mkt_ask}"
        
        return xr

    def swap(self):
        amm_ask, amm_bid, mkt_ask, mkt_bid = self.oracle.get_price()
        x_r = self._calculate_swap_amount(amm_ask, amm_bid, mkt_ask, mkt_bid)
        swap_info = self.amm.swap(x_r)
        swap_info.update(self._calculate_trade_metrics(swap_info, amm_ask, amm_bid))
        return swap_info

    def _calculate_trade_metrics(self, swap_info, prev_amm_ask, prev_amm_bid):
        amm_ask, amm_bid, mkt_ask, mkt_bid = self.oracle.get_price()
        pr_bid, pr_ask, pr_mid = self.oracle.get_token_prices('r').values()
        ps_bid, ps_ask, ps_mid = self.oracle.get_token_prices('s').values()
        initial_cash = self.amm.initial_ls * self.oracle.initial_price * 2
        fee_cost = (swap_info['token_fee']['r'] * self.oracle.get_token_prices('r')['ask'] + 
                    swap_info['token_fee']['s'] * self.oracle.get_token_prices('s')['ask'])
        amm_cost = (swap_info['arbitrage_cost']['r'] * self.oracle.get_token_prices('r')['ask'] +
                    swap_info['arbitrage_cost']['s'] * self.oracle.get_token_prices('s')['ask'])
        mkt_gain = (swap_info['arbitrage_gain']['r'] * self.oracle.get_token_prices('r')['bid'] +
                    swap_info['arbitrage_gain']['s'] * self.oracle.get_token_prices('s')['bid'])
        pnl = mkt_gain - amm_cost

        self.pnl += pnl
        if swap_info['arbitrage_gain'] != 0:
            self.total_number_trade += 1
            self.total_fee += fee_cost

        initial_pool_value = (self.amm.initial_lr * self.oracle.get_token_prices('r')['mid'] + 
                            self.amm.initial_ls * self.oracle.get_token_prices('s')['mid'])
        current_pool_value = (self.amm.lr * self.oracle.get_token_prices('r')['mid'] + 
                            self.amm.ls * self.oracle.get_token_prices('s')['mid'])
        
        return {
            'fee_dollar_value': fee_cost,
            'total_fee_dollar_value': self.total_fee,
            'mkt_gain': mkt_gain,
            'amm_cost': amm_cost,
            'trader_step_pnl': pnl,
            'trader_total_pnl': self.pnl,
            'initial_pool_value': initial_pool_value,
            'current_pool_value': current_pool_value,
            'impermanent_loss': current_pool_value - initial_pool_value,
            'net_profit': self.total_fee + (current_pool_value - initial_pool_value),
            'account_profit': self.total_fee + (current_pool_value - initial_cash),
            'mkt_ask': mkt_ask,
            'mkt_bid': mkt_bid,
            'prev_amm_ask': prev_amm_ask,
            'prev_amm_bid': prev_amm_bid,
            'amm_ask': amm_ask,
            'amm_bid': amm_bid,
            'mid_r': pr_mid,
            'mid_s': ps_mid,
            'bid_r': pr_bid,
            'bid_s': ps_bid,
            'ask_r': pr_ask,
            'ask_s': ps_ask,
            'total_number_trade': self.total_number_trade,
            'spread': self.oracle.spread,
            'fee_rate': self.amm.f
        }