
class AMM:
    def __init__(self, initial_ls=1000000, initial_lr=1000000, fee_rate=0.003, fee_source=1):
        """
        Automated Market Maker (AMM) with constant product formula and configurable fee structure.
   
        Parameters:
            initial_ls (int): Initial stable token liquidity
            initial_lr (int): Initial risky token liquidity
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
        self.initial_ls = initial_ls
        self.initial_lr = initial_lr
        self.ls = initial_ls
        self.lr = initial_lr
        self.k = self.ls * self.lr
        self.f = fee_rate
        self.fee_source = fee_source
        self.reset()
    
    def reset(self, fee_source=None):
        self.lr = self.initial_lr
        self.ls = self.initial_ls
        self.fee = {'r': 0, 's': 0}
        self.fee_source = fee_source if fee_source is not None else self.fee_source

    def get_price(self):
        amm_ask = (self.ls / self.lr) / (1 - self.f)
        amm_bid = (self.ls / self.lr) * (1 - self.f)
        return amm_ask, amm_bid

    def _handle_positive_xr(self, xr):

        if self.fee_source > 0:
            xs = (self.ls * self.lr) / (self.lr + (1-self.f)*xr) - self.ls
            self.lr += xr * (1-self.f)
            self.ls += xs
            fee = xr * self.f
            step_fee = {'r': fee, 's': 0}
            arbitrage_gain = {'r': 0, 's': -xs}
            arbitrage_cost = {'r': xr, 's': 0}
        else:
            xs = (self.ls * self.lr) / (self.lr + xr) - self.ls
            self.lr += xr
            self.ls += xs
            fee = -xs * self.f
            step_fee = {'r': 0, 's': fee}
            arbitrage_gain = {'r': 0, 's': -xs * (1-self.f)}
            arbitrage_cost = {'r': xr, 's': 0}
        
        return arbitrage_gain, arbitrage_cost, step_fee

    def _handle_negative_xr(self, xr):
        if self.fee_source > 0:
            xs = ((self.ls * self.lr) / (self.lr + xr) - self.ls) / (1-self.f)
            self.lr += xr
            self.ls += xs * (1-self.f)
            fee = xs * self.f
            step_fee = {'r': 0, 's': fee}
            arbitrage_gain = {'r': -xr, 's': 0}
            arbitrage_cost = {'r': 0, 's': xs} # include fee
        else:
            xs = (self.ls * self.lr) / (self.lr + xr) - self.ls
            self.lr += xr
            self.ls += xs
            fee = -xr * self.f
            step_fee = {'r': fee, 's': 0}
            arbitrage_gain = {'r': -xr * (1-self.f), 's': 0}
            arbitrage_cost = {'r': 0, 's': xs}
        
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