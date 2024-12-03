import numpy as np
from env.amm import AMM

class OracleSimulator:
    def __init__(self, fee_rate=0.003, fee_distribute=True, fee_source=1,
                 initial_lr=1000000, initial_ls=1000000, start_price=500,
                 mu=0, sigma=0.2, dt=1/(252*6.5*60), steps=23400,
                 spread=0, alpha=0.5, seed=None):
        
        self.amm = AMM(fee_rate=fee_rate, initial_lr=initial_lr, 
                       initial_ls=initial_ls, fee_distribute=fee_distribute, 
                       fee_source=fee_source)
        
        self.rng = np.random.default_rng(seed)
        self.initial_price = start_price
        self.alpha = alpha
        self.mu = mu
        self.random_sigma = sigma < 0
        self.sigma = sigma if sigma >= 0 else self.get_random_sigma()
        self.spread = spread
        self.dt = dt
        self.steps = steps
        self.reset()

    def get_random_sigma(self):
        return self.rng.uniform(0.05, 0.35)

    def get_token_prices(self, token):
        if token not in ["s", "r"]:
            raise ValueError("Invalid token. Must be 's' or 'r'")
            
        price = self.ps if token == "s" else self.pr
        return {
            'bid': price - self.spread,
            'ask': price + self.spread,
            'mid': price
        }

    def get_price(self):
        amm_ask, amm_bid = self.amm.get_price()
        s_prices = self.get_token_prices('s')
        r_prices = self.get_token_prices('r')
        
        mkt_ask = r_prices['ask'] / s_prices['bid']
        mkt_bid = r_prices['bid'] / s_prices['ask']
        
        return amm_ask, amm_bid, mkt_ask, mkt_bid

    def next(self):
        self.index += 1
        gbm_factor = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        random_factor = self.sigma * np.sqrt(self.dt) * self.rng.normal()
        
        self.ps *= np.exp(gbm_factor + random_factor)
        self.pr *= np.exp(gbm_factor + random_factor)

    def reset(self):
        self.amm.reset()
        self.ps = self.initial_price
        self.pr = self.ps * (self.amm.ls / self.amm.lr) * (1 - self.amm.f)**(2*self.alpha-1)
        self.index = 0