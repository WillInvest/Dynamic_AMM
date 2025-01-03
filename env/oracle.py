import numpy as np
from env.amm import AMM

class OracleSimulator:
    def __init__(self, amm, start_price=500,
                 mu=0, sigma=0.2, dt=1/(252*6.5*60), steps=23400,
                 spread=0, alpha=0.5, kappa=1.0, seed=None):
        """
        Simulates oracle prices for two tokens (stable and risky) and manages AMM interaction.
   
        Parameters:
            fee_rate (float): AMM fee percentage, default 0.003 (0.3%)
            fee_distribute (bool): If True, distributes fees to LPs immediately, default True
            fee_source (int): 1 to collect fees from incoming tokens, -1 from outgoing tokens
            initial_lr (int): Initial risky token liquidity in AMM, default 1M
            initial_ls (int): Initial stable token liquidity in AMM, default 1M 
            start_price (float): Initial price of stable token, default 500
            mu (float): Drift term for price evolution, default 0
            sigma (float): Volatility factor for price evolution, default 0.2
                          If negative, randomly samples from U(0.05, 0.35)
            dt (float): Time step size, default 1/(252*6.5*60) (1 second in trading days)
            steps (int): Number of simulation steps, default 23400 (1 trading day)
            spread (float): Market maker spread added to oracle token prices, default 0
            alpha (float): Position of initial oracle price relative to AMM spread
                          0.5 aligns with AMM mid price
                          0 aligns with AMM bid price  
                          1 aligns with AMM ask price
            kappa (float): Mean reversion factor for stable token price, default 1.0
            seed (int): Random number generator seed for reproducibility, default None
        """
        
        self.amm = amm
        self.rng = np.random.default_rng(seed)
        self.initial_price = start_price
        self.alpha = alpha
        self.kappa = kappa
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
        s_drift = self.kappa * (1 - self.ps/self.initial_price)
        
        # Separate random draws
        z1 = self.rng.normal()
        z2 = self.rng.normal()
        
        self.ps *= np.exp(s_drift * self.dt + (self.sigma * 0.1) * np.sqrt(self.dt) * z1)
        self.pr *= np.exp(self.mu * self.dt + self.sigma * np.sqrt(self.dt) * z2)

    def reset(self):
        self.amm.reset()
        self.ps = self.initial_price
        self.pr = self.ps * (self.amm.ls / self.amm.lr) * (1 - self.amm.f)**(2*self.alpha-1)
        self.index = 0