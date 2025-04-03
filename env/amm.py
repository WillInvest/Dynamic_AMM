import numpy as np
class AMM:
    def __init__(self, x0=1000, y0=1000, fee_rate=0.003, fee_source=1):
        """
        Automated Market Maker (AMM) with constant product formula and configurable fee structure.
   
        Parameters:
            x0 (int): Initial stable token liquidity
            y0 (int): Initial risky token liquidity
            fee_rate (float): Trading fee percentage (e.g., 0.003 for 0.3%)
            fee_distribute (bool): If True, fees are added to liquidity pool immediately
            fee_source (int): 1 to charge fees on input tokens, -1 for output tokens
   
        Methods:
            reset(): Resets AMM state to initial values
            get_price(): Returns AMM ask/bid prices 
            swap(xr): Executes token swap based on xr amount
                - xr > 0: Swap risky tokens for stable tokens
                - xr < 0: Swap stable tokens for risky tokens
                Returns dict with swap details including:
                - Amounts swapped (xs, xr)
                - Pre/post liquidity levels
                - Fees collected
           
        Private Methods:
            _handle_positive_xr(): Handles swaps with positive xr (r->s)
            _handle_negative_xr(): Handles swaps with negative xr (s->r)
        """
        self.x0 = x0
        self.y0 = y0
        self.x = x0
        self.y = y0
        self.k = self.x * self.y
        self.gamma = fee_rate
        self.fee_source = fee_source
        self.reset()
    
    def reset(self, fee_source=None):
        self.x = self.x0
        self.y = self.y0
        self.L = np.sqrt(self.x * self.y)
        self.fee = {'r': 0, 's': 0}
        self.fee_source = fee_source if fee_source is not None else self.fee_source

    def get_price(self):
        amm_ask = (self.x / self.y) / (1 - self.gamma)
        amm_bid = (self.x / self.y) * (1 - self.gamma)
        return amm_ask, amm_bid

    def _handle_positive_dr(self, dr):

        if self.fee_source > 0:
            ds = (self.x * self.y) / (self.y + (1-self.gamma)*dr) - self.x
            self.y += dr * (1-self.gamma)
            self.x += ds
            fee = dr * self.gamma
            step_fee = {'r': fee, 's': 0}
            arbitrage_gain = {'r': 0, 's': -ds}
            arbitrage_cost = {'r': dr, 's': 0}
        else:
            xs = (self.x * self.y) / (self.y + dr) - self.x
            self.y += dr
            self.x += xs
            fee = -xs * self.gamma
            step_fee = {'r': 0, 's': fee}
            arbitrage_gain = {'r': 0, 's': -xs * (1-self.gamma)}
            arbitrage_cost = {'r': dr, 's': 0}
        
        return arbitrage_gain, arbitrage_cost, step_fee

    def _handle_negative_dr(self, dr):
        if self.fee_source > 0:
            ds = ((self.x * self.y) / (self.y + dr) - self.x) / (1-self.gamma)
            self.y += dr
            self.x += ds * (1-self.gamma)
            fee = ds * self.gamma
            step_fee = {'r': 0, 's': fee}
            arbitrage_gain = {'r': -dr, 's': 0}
            arbitrage_cost = {'r': 0, 's': ds} # include fee
        else:
            ds = ((self.x * self.y) / (self.y + dr) - self.x) / (1-self.gamma)
            self.y += dr
            self.x += ds
            fee = -dr * self.gamma
            step_fee = {'r': fee, 's': 0}
            arbitrage_gain = {'r': -dr * (1-self.gamma), 's': 0}
            arbitrage_cost = {'r': 0, 's': ds}
        
        return arbitrage_gain, arbitrage_cost, step_fee

    def swap(self, xr):
        assert self.f >= 0, "Fee rate must be non-negative"
        pre_lr, pre_ls = self.lr, self.ls
        
        if xr > 0:
            arbitrage_gain, arbitrage_cost, step_fee = self._handle_positive_xr(xr)
        elif xr < 0:
            arbitrage_gain, arbitrage_cost, step_fee = self._handle_negative_xr(xr)
        else:
            arbitrage_gain, arbitrage_cost, step_fee = {'r': 0, 's': 0}, {'r': 0, 's': 0}, {'r': 0, 's': 0}

        return {
            'pre_ls': pre_ls,
            'pre_lr': pre_lr,
            'prev_k': pre_ls * pre_lr,
            'ls': self.ls,
            'lr': self.lr,
            'current_k': self.ls * self.lr,
            'token_fee': step_fee,
            'arbitrage_gain': arbitrage_gain,
            'arbitrage_cost': arbitrage_cost
        }