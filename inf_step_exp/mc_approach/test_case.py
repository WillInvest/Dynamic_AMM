import numpy as np
import pandas as pd

class Test_Simulator:
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
        # Set the seed for reproducibility
        rng = np.random.default_rng(self.seed)
        
        # Generate random normal variables
        Z = rng.normal(0, 1, size=(self.steps,))
        
        # Initialize price array
        S = np.zeros((self.steps + 1,))
        S[0] = self.s0
        
        # Calculate time points and terms
        t = np.arange(1, self.steps + 1) * self.dt
        drift_term = (self.drift - 0.5 * self.sigma**2) * t
        diffusion_term = self.sigma * np.sqrt(self.dt) * np.cumsum(Z)
        
        # Calculate final prices
        S[1:] = self.s0 * np.exp(drift_term + diffusion_term)
            
        return S
    
    def simulate(self) -> pd.DataFrame: # simulate the paths and return a dataframe
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

            # distribute case
            dis_p = dis_y[i] / dis_x[i]
            dis_upper_bound = dis_p / (1 - self.gamma)
            dis_lower_bound = dis_p * (1 - self.gamma)
            dis_L = np.sqrt(dis_y[i] * dis_x[i])
            assert abs(dis_L - self.L) < 1e-8, f"distribute case, dis_L: {dis_L}, self.L: {self.L}" 
            if s > dis_upper_bound:
                dis_x[i+1] = dis_L / np.sqrt((1-self.gamma) * s)
                dis_y[i+1] = dis_L * np.sqrt((1-self.gamma) * s)
                dis_in_fee[i+1] = dis_in_fee[i] + self.gamma/(1-self.gamma) * (dis_y[i+1] - dis_y[i])
                dis_out_fee[i+1] = dis_out_fee[i] + self.gamma * s * (dis_x[i] - dis_x[i+1])
                assert dis_in_fee[i+1] < dis_out_fee[i+1], f"distribute case, dis_in_fee: {dis_in_fee[i+1]}, dis_out_fee: {dis_out_fee[i+1]}"
                
            elif s < dis_lower_bound:
                dis_x[i+1] = dis_L * np.sqrt((1-self.gamma) / s)
                dis_y[i+1] = dis_L * np.sqrt(s / (1-self.gamma))
                dis_in_fee[i+1] = dis_in_fee[i] + self.gamma/(1-self.gamma) * s * (dis_x[i+1] - dis_x[i])
                dis_out_fee[i+1] = dis_out_fee[i] + self.gamma * (dis_y[i] - dis_y[i+1])
                assert dis_in_fee[i+1] < dis_out_fee[i+1], f"distribute case, dis_in_fee: {dis_in_fee[i+1]}, dis_out_fee: {dis_out_fee[i+1]}"
            else:
                dis_x[i+1] = dis_x[i]
                dis_y[i+1] = dis_y[i]
                dis_in_fee[i+1] = dis_in_fee[i]
                dis_out_fee[i+1] = dis_out_fee[i]
            
            # rebalance case
            re_in_p = re_in_y[i] / re_in_x[i]
            re_out_p = re_out_y[i] / re_out_x[i]
            
            # incoming case
            re_in_upper_bound = re_in_p / (1 - self.gamma)
            re_in_lower_bound = re_in_p * (1 - self.gamma)
            if s > re_in_upper_bound:
                re_in_a = 1 - self.gamma
                re_in_b = (2-self.gamma) * re_in_y[i]
                re_in_c = re_in_y[i]**2 - re_in_L**2 * s * (1-self.gamma)
                delta_y_re_in = (-re_in_b + np.sqrt(re_in_b**2 - 4 * re_in_a * re_in_c)) / (2 * re_in_a)
                re_in_y[i+1] = re_in_y[i] + delta_y_re_in
                re_in_x[i+1] = re_in_y[i+1] / (s * (1-self.gamma))
                assert np.sqrt(re_in_x[i+1] * re_in_y[i+1]) > re_in_L, f"rebalance incoming case, re_in_L: {re_in_L}, new_re_in_L: {np.sqrt(re_in_x[i+1] * re_in_y[i+1])}"
                re_in_L = np.sqrt(re_in_x[i+1] * re_in_y[i+1])
                
            elif s < re_in_lower_bound:
                re_in_a = 1 - self.gamma
                re_in_b = (2-self.gamma) * re_in_x[i]
                re_in_c = re_in_x[i]**2 - re_in_L**2 * (1-self.gamma) / s
                delta_x_re_in = (-re_in_b + np.sqrt(re_in_b**2 - 4 * re_in_a * re_in_c)) / (2 * re_in_a)
                re_in_x[i+1] = re_in_x[i] + delta_x_re_in
                re_in_y[i+1] = re_in_x[i+1] * s / (1-self.gamma)
                assert np.sqrt(re_in_x[i+1] * re_in_y[i+1]) > re_in_L, f"rebalance incoming case, re_in_L: {re_in_L}, new_re_in_L: {np.sqrt(re_in_x[i+1] * re_in_y[i+1])}"
                re_in_L = np.sqrt(re_in_x[i+1] * re_in_y[i+1])
                
            else:
                re_in_x[i+1] = re_in_x[i]
                re_in_y[i+1] = re_in_y[i]
                
            # outgoing case
            re_out_upper_bound = re_out_p / (1 - self.gamma)
            re_out_lower_bound = re_out_p * (1 - self.gamma)
            if s > re_out_upper_bound:
                re_out_a = 1 - self.gamma
                re_out_b = -(2-self.gamma) * re_out_x[i]
                re_out_c = re_out_x[i]**2 - re_out_L**2 / (s * (1-self.gamma))
                delta_x_re_out = (-re_out_b - np.sqrt(re_out_b**2 - 4 * re_out_a * re_out_c)) / (2 * re_out_a)
                re_out_x[i+1] = re_out_x[i] - (1-self.gamma) * delta_x_re_out
                re_out_y[i+1] = re_out_x[i+1] * s * (1-self.gamma)
                assert np.sqrt(re_out_x[i+1] * re_out_y[i+1]) > re_out_L, f"rebalance outgoing case, re_out_L: {re_out_L}, new_re_out_L: {np.sqrt(re_out_x[i+1] * re_out_y[i+1])}"
                re_out_L = np.sqrt(re_out_x[i+1] * re_out_y[i+1])
                
            elif s < re_out_lower_bound:
                re_out_a = 1 - self.gamma
                re_out_b = -(2-self.gamma) * re_out_y[i]
                re_out_c = re_out_y[i]**2 - re_out_L**2 * s / (1-self.gamma)
                delta_y_re_out = (-re_out_b - np.sqrt(re_out_b**2 - 4 * re_out_a * re_out_c)) / (2 * re_out_a)
                re_out_y[i+1] = re_out_y[i] - (1-self.gamma) * delta_y_re_out
                re_out_x[i+1] = re_out_y[i+1] * (1-self.gamma) / s
                assert np.sqrt(re_out_x[i+1] * re_out_y[i+1]) > re_out_L, f"rebalance outgoing case, re_out_L: {re_out_L}, new_re_out_L: {np.sqrt(re_out_x[i+1] * re_out_y[i+1])}"
                re_out_L = np.sqrt(re_out_x[i+1] * re_out_y[i+1])
            else:
                re_out_x[i+1] = re_out_x[i]
                re_out_y[i+1] = re_out_y[i]
                
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
            
        df = pd.DataFrame(results)
                
        return df
        
    

if __name__ == "__main__":
    simulator = Test_Simulator()
    paths = simulator.generate_paths()
    print(paths)
    
    