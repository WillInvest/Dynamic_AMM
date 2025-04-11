import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def generate_paths_jit(s0, drift, sigma, dt, steps, seed):
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Generate random normal variables
    Z = np.random.normal(0, 1, size=(steps,))
    
    # Initialize price array
    S = np.zeros((steps + 1,))
    S[0] = s0
    
    # Calculate time points and terms
    t = np.arange(1, steps + 1) * dt
    drift_term = (drift - 0.5 * sigma**2) * t
    diffusion_term = sigma * np.sqrt(dt) * np.cumsum(Z)
    
    # Calculate final prices
    S[1:] = s0 * np.exp(drift_term + diffusion_term)
        
    return S

@jit(nopython=True)
def simulate_distribute_case(x, y, s, gamma, L):
    dis_p = y / x
    dis_upper_bound = dis_p / (1 - gamma)
    dis_lower_bound = dis_p * (1 - gamma)
    
    if s > dis_upper_bound:
        new_x = L / np.sqrt((1-gamma) * s)
        new_y = L * np.sqrt((1-gamma) * s)
        in_fee = gamma/(1-gamma) * (new_y - y)
        out_fee = gamma * s * (x - new_x)
    elif s < dis_lower_bound:
        new_x = L * np.sqrt((1-gamma) / s)
        new_y = L * np.sqrt(s / (1-gamma))
        in_fee = gamma/(1-gamma) * s * (new_x - x)
        out_fee = gamma * (y - new_y)
    else:
        new_x = x
        new_y = y
        in_fee = 0.0
        out_fee = 0.0
    
    return new_x, new_y, in_fee, out_fee

@jit(nopython=True)
def simulate_rebalance_case(x, y, s, gamma, L, is_incoming=True):
    p = y / x
    upper_bound = p / (1 - gamma)
    lower_bound = p * (1 - gamma)
    
    new_x = x
    new_y = y
    
    if is_incoming:
        if s > upper_bound:
            a = 1 - gamma
            b = (2-gamma) * y
            c = y**2 - L**2 * s * (1-gamma)
            delta_y = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            new_y = y + delta_y
            new_x = new_y / (s * (1-gamma))
        elif s < lower_bound:
            a = 1 - gamma
            b = (2-gamma) * x
            c = x**2 - L**2 * (1-gamma) / s
            delta_x = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            new_x = x + delta_x
            new_y = new_x * s / (1-gamma)
    else:  # outgoing case
        if s > upper_bound:
            a = 1 - gamma
            b = -(2-gamma) * x
            c = x**2 - L**2 / (s * (1-gamma))
            delta_x = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            new_x = x - (1-gamma) * delta_x
            new_y = new_x * s * (1-gamma)
        elif s < lower_bound:
            a = 1 - gamma
            b = -(2-gamma) * y
            c = y**2 - L**2 * s / (1-gamma)
            delta_y = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            new_y = y - (1-gamma) * delta_y
            new_x = new_y * (1-gamma) / s
    
    return new_x, new_y

class Test_Simulator_JIT:
    def __init__(self, x=1000.0, y=1000.0, gamma=0.003, s0=1.0, drift=0.0, sigma=0.2, 
                 dt=1/365, steps=10000, seed=0):
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.L = np.sqrt(x * y)
        self.gamma = gamma
        self.s0 = s0
        self.drift = drift
        self.sigma = sigma
        self.dt = dt
        self.steps = steps
        self.seed = seed
        
    def generate_paths(self):
        return generate_paths_jit(self.s0, self.drift, self.sigma, self.dt, self.steps, self.seed)
    
    def simulate(self) -> pd.DataFrame:
        results = []
        S = self.generate_paths()
        dis_x = np.zeros((self.steps+1,), dtype=np.float64)
        dis_y = np.zeros((self.steps+1,), dtype=np.float64)
        re_in_x = np.zeros((self.steps+1,), dtype=np.float64)
        re_in_y = np.zeros((self.steps+1,), dtype=np.float64)
        re_out_x = np.zeros((self.steps+1,), dtype=np.float64)
        re_out_y = np.zeros((self.steps+1,), dtype=np.float64)
        dis_in_fee = np.zeros((self.steps+1,), dtype=np.float64)
        dis_out_fee = np.zeros((self.steps+1,), dtype=np.float64)
        
        # Initialize values
        dis_x[0] = self.x
        dis_y[0] = self.y
        re_in_x[0] = self.x
        re_in_y[0] = self.y
        re_out_x[0] = self.x
        re_out_y[0] = self.y
        re_in_L = np.sqrt(self.x * self.y)
        re_out_L = np.sqrt(self.x * self.y)
        
        results.append({
            'time_step': 0,
            'price': S[0],
            'dis_x': dis_x[0],
            'dis_y': dis_y[0],
            're_in_x': re_in_x[0],
            're_in_y': re_in_y[0],
            're_out_x': re_out_x[0],
            're_out_y': re_out_y[0],
            'dis_in_fee': dis_in_fee[0],
            'dis_out_fee': dis_out_fee[0],
            're_in_L': re_in_L,
            're_out_L': re_out_L,
            'dis_L': self.L,
        })
        
        for i in range(self.steps):
            s = S[i+1]
            
            # Distribute case
            dis_x[i+1], dis_y[i+1], in_fee, out_fee = simulate_distribute_case(
                dis_x[i], dis_y[i], s, self.gamma, self.L
            )
            dis_in_fee[i+1] = dis_in_fee[i] + in_fee
            dis_out_fee[i+1] = dis_out_fee[i] + out_fee
            
            # Rebalance incoming case
            re_in_x[i+1], re_in_y[i+1] = simulate_rebalance_case(
                re_in_x[i], re_in_y[i], s, self.gamma, re_in_L, is_incoming=True
            )
            if re_in_x[i+1] != re_in_x[i] or re_in_y[i+1] != re_in_y[i]:
                re_in_L = np.sqrt(re_in_x[i+1] * re_in_y[i+1])
            
            # Rebalance outgoing case
            re_out_x[i+1], re_out_y[i+1] = simulate_rebalance_case(
                re_out_x[i], re_out_y[i], s, self.gamma, re_out_L, is_incoming=False
            )
            if re_out_x[i+1] != re_out_x[i] or re_out_y[i+1] != re_out_y[i]:
                re_out_L = np.sqrt(re_out_x[i+1] * re_out_y[i+1])
            
            dis_L = np.sqrt(dis_y[i+1] * dis_x[i+1])
            assert abs(dis_L - self.L) < 1e-8, f"distribute case, dis_L: {dis_L}, self.L: {self.L}"
            
            results.append({
                'time_step': i+1,
                'price': s,
                'dis_x': dis_x[i+1],
                'dis_y': dis_y[i+1],
                're_in_x': re_in_x[i+1],
                're_in_y': re_in_y[i+1],
                're_out_x': re_out_x[i+1],
                're_out_y': re_out_y[i+1],
                'dis_in_fee': dis_in_fee[i+1],
                'dis_out_fee': dis_out_fee[i+1],
                're_in_L': re_in_L,
                're_out_L': re_out_L,
                'dis_L': dis_L
            })
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Use the same parameters as in test.ipynb
    sigma = 0.1
    gamma = 0.002
    dt = 1/(365 * 24)
    steps = 100
    seed = 0
    
    simulator = Test_Simulator_JIT(sigma=sigma, gamma=gamma, dt=dt, steps=steps, seed=seed)
    df = simulator.simulate()
    print(df.to_markdown()) 