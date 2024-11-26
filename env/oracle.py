import os
import sys
# Get the path to the Dynamic_AMM directory
sys.path.append(f'{os.path.expanduser("~")}/Dynamic_AMM')
import numpy as np
from amm import AMM

class OracleSimulator:
    def __init__(self,
                 fee_rate=0.003,
                 fee_distribute=True,
                 fee_source=1,
                 initial_lr=1000000,
                 initial_ls=1000000,
                 start_price=500,
                 mu=0,
                 sigma=0.2,
                 dt=1/(252*6.5*60),  # 1 second in trading days
                 steps=23400,  # number of seconds in 20 trading days
                 spread=0,
                 alpha=0.5,
                 seed=None):
        
        self.amm = AMM(fee_rate=fee_rate, initial_lr=initial_lr, initial_ls=initial_ls, fee_distribute=fee_distribute, fee_source=fee_source)
        
        # Initialize random number generator, only if seed is provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()  # Random behavior without a seed
        self.initial_price = start_price
        self.alpha = alpha
        self.reset()
        self.mu = mu
        self.random_sigma = True if sigma<0 else False
        self.sigma = sigma if sigma>=0 else self.get_random_sigma()
        self.spread = spread
        self.dt = dt
        self.index = 0
        self.steps = steps

    def get_random_sigma(self):
        # Sample a sigma value
        sampled_sigma = self.rng.uniform(0.05, 0.35)
        return sampled_sigma

    def get_bid_price(self, token):
        if token == "s":
            return self.ps - self.spread
        elif token == "r":
            return self.pr - self.spread
        else:
            print("Invalid input")
            
    def get_ask_price(self, token):
        if token == "s":
            return self.ps + self.spread
        elif token == "r":
            return self.pr + self.spread
        else:
            print("Invalid input")
            
    def get_mid_price(self, token):
        if token == "s":
            return self.ps
        elif token == "r":
            return self.pr
        else:
            print("Invalid input")
            
    def get_price(self):
        amm_ask, amm_bid = self.amm.get_price()
        mkt_ask = self.get_ask_price('r') / self.get_bid_price('s')
        mkt_bid = self.get_bid_price('r') / self.get_ask_price('s')
        return amm_ask, amm_bid, mkt_ask, mkt_bid

    def next(self):
        # update the index
        self.index += 1
        
        # Update the current price using the GBM formula with the determined sigma
        self.ps *= np.exp(  
            (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.rng.normal())
        self.pr *= np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * self.rng.normal())
        
    def reset(self):
        self.amm.reset()
        self.ps = self.initial_price
        self.pr = self.ps * (self.amm.ls / self.amm.lr) * (1 - self.amm.f)**(2*self.alpha-1)
        self.index = 0
        

    
    
   