import numpy as np
import random

class AMM:
    def __init__(self, ls=1000000, lr=1000000, fee=0.003, distribute=True):
        self.initial_ls = ls
        self.initial_lr = lr
        self.ls = ls
        self.lr = lr
        self.k = self.ls * self.lr
        self.f = fee 
        self.distribute = distribute

    def get_price(self):
        amm_ask = (self.ls / self.lr) / (1-self.f)
        amm_bid = (self.ls / self.lr) * (1-self.f)
        return amm_ask, amm_bid

    def swap(self, xr):
        assert self.f >= 0, "Fee rate must be non-negative"        
        pre_lr = self.lr
        pre_ls = self.ls
        # Calculate the amount of asset to be swapped
        if xr > 0:
            xs = (self.ls * self.lr) / (self.lr + (1-self.f)*xr) - self.ls # negative and swap out 
            fee = xr * self.f
            # add the fee to the fee pool if fee_pool is True
            if not self.distribute:
                self.lr += fee 
                fee = 0
            xr *= (1-self.f)
            self.lr += xr
            self.ls += xs
        elif xr < 0:
            xs = (1/(1-self.f)) * ((self.ls * self.lr) / (self.lr + xr) - self.ls)
            fee = xs * self.f
            # add the fee to the fee pool if fee_pool is True
            if not self.distribute:
                self.ls += fee
                fee = 0
            xs *= (1-self.f)
            self.ls += xs
            self.lr += xr
        else:
            xs = 0
            fee = 0

        # return the information of the swap
        info = {'xs': xs, 'xr': xr, 'pre_ls': pre_ls, 'pre_lr': pre_lr,
                'ls':self.ls, 'lr':self.lr, 'token_fee': fee}
        return info

    def reset(self):
        self.lr = self.initial_lr
        self.ls = self.initial_ls

    
if __name__ == '__main__':
    import math
    initial_a = 8000
    initial_b = 10000
    amm = AMM(initial_a, initial_b, fee=0.1)
    info = amm.swap(-100)
    
    print(info)
    
    print(f"previous k = {initial_a * initial_b}")
    print(f"new k = {info['ls'] * info['lr']}")

    
    

    
    
    
    
    