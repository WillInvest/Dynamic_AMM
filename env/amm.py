import numpy as np

class AMM:
    def __init__(self, initial_ls=1000000, initial_lr=1000000, fee_rate=0.003, 
                 fee_distribute=True, fee_source=1):
        self.initial_ls = initial_ls
        self.initial_lr = initial_lr
        self.ls = initial_ls
        self.lr = initial_lr
        self.k = self.ls * self.lr
        self.f = fee_rate
        self.fee_distribute = fee_distribute
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
            fee = xr * self.f
            step_fee = {'r': fee, 's': 0}
            self.fee['r'] += fee
            xr *= (1-self.f)
        else:
            xs = (self.ls * self.lr) / (self.lr + xr) - self.ls
            fee = -xs * self.f
            step_fee = {'r': 0, 's': fee}
            self.fee['s'] += fee
        
        self.lr += xr
        self.ls += xs
        return xs, step_fee

    def _handle_negative_xr(self, xr):
        if self.fee_source > 0:
            xs = (1/(1-self.f)) * ((self.ls * self.lr) / (self.lr + xr) - self.ls)
            fee = xs * self.f
            step_fee = {'r': 0, 's': fee}
            self.fee['s'] += fee
            xs *= (1-self.f)
        else:
            xs = (self.ls * self.lr) / (self.lr + xr) - self.ls
            fee = -xr * self.f
            step_fee = {'r': fee, 's': 0}
            self.fee['r'] += fee
        
        self.ls += xs
        self.lr += xr
        return xs, step_fee

    def swap(self, xr):
        assert self.f >= 0, "Fee rate must be non-negative"
        pre_lr, pre_ls = self.lr, self.ls
        
        if xr > 0:
            xs, step_fee = self._handle_positive_xr(xr)
        elif xr < 0:
            xs, step_fee = self._handle_negative_xr(xr)
        else:
            xs, step_fee = 0, {'r': 0, 's': 0}

        return {
            'xs': xs, 
            'xr': xr,
            'pre_ls': pre_ls,
            'pre_lr': pre_lr,
            'ls': self.ls,
            'lr': self.lr,
            'token_fee': step_fee
        }