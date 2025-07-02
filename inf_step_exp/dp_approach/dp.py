# %%

import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *

class DPAMM:
    def __init__(self, nS, nY, nG, T,
                 L, gamma, sigma, dt,
                 smax=1.5, smin=0.5,
                 ymax=1.5, ymin=0.5,
                 gmax=0.01, gmin=0.0005,
                 price_dynamics='GBM'):
        
        self.nS = nS
        self.nY = nY
        self.nG = nG
        self.T = T
        self.L = L
        self.gamma = gamma
        self.sigma = sigma
        self.dt = dt
        self.price_dynamics = price_dynamics
        self.dynamic_fee = True if gamma is None else False
        self.best_gamma = np.zeros((self.nS, self.nY)) if self.dynamic_fee else None
        
        self.init_grid(smax, smin, ymax, ymin, gmax, gmin)
        self.init_transition_probs()
        self.init_value_function()
        
    def init_grid(self, smax, smin, ymax, ymin, gmax, gmin):
        self.S_grid = np.linspace(smin, smax, self.nS)
        self.Y_grid = np.linspace(ymin, ymax, self.nY)
        self.gamma_grid = np.linspace(gmin, gmax, self.nG)
        if self.dynamic_fee:
            self.S_mesh, self.Y_mesh, self.G_mesh =\
                np.meshgrid(self.S_grid, self.Y_grid, self.gamma_grid, indexing='ij')
        else:
            self.S_mesh, self.Y_mesh =\
                np.meshgrid(self.S_grid, self.Y_grid, indexing='ij')
        self.fee = np.zeros_like(self.S_mesh)
        self.y_next = np.zeros_like(self.Y_mesh)
        self.cont_value = np.zeros_like(self.S_mesh)
        
    def init_transition_probs(self):
        def transition_prob(s0, s1):
            if self.price_dynamics == 'GBM':
                return (1 / (s1 * self.sigma * np.sqrt(2 * np.pi * self.dt))) * \
                    np.exp(- (np.log(s1) - np.log(s0))**2 / (2 * self.sigma**2 * self.dt))
            else:
                raise ValueError(f"Invalid price dynamics: {self.price_dynamics}")
        self.transition_probs = np.zeros((self.nS, self.nS))
        for i in range(self.nS):
            for j in range(self.nS):
                self.transition_probs[i, j] = transition_prob(self.S_grid[i], self.S_grid[j])
            self.transition_probs[i] /= self.transition_probs[i].sum()
        
    def init_value_function(self):
        self.V = np.zeros((self.nS, self.nY))
        for i in range(self.nS):
            for j in range(self.nY):
                self.V[i, j] = (self.L**2 * self.S_grid[i] + self.Y_grid[j]**2) / self.Y_grid[j]
                
    def find_condition(self):
        P_mesh = self.Y_mesh**2 / self.L**2
        if self.dynamic_fee:
            self.cond1 = self.S_mesh > P_mesh / (1 - self.G_mesh)
            self.cond2 = self.S_mesh < P_mesh * (1 - self.G_mesh)
            self.cond3 = ~(self.cond1 | self.cond2)
        else:
            self.cond1 = self.S_mesh > P_mesh / (1 - self.gamma)
            self.cond2 = self.S_mesh < P_mesh * (1 - self.gamma)
            self.cond3 = ~(self.cond1 | self.cond2)
            
    def update_fee(self):
        if self.dynamic_fee:
            self.fee[self.cond1] = self.G_mesh[self.cond1] / (1 - self.G_mesh[self.cond1]) *\
                (self.L * np.sqrt((1 - self.G_mesh[self.cond1]) * self.S_mesh[self.cond1]) - self.Y_mesh[self.cond1])
            self.fee[self.cond2] = self.G_mesh[self.cond2] / (1 - self.G_mesh[self.cond2]) *\
                (self.L * np.sqrt((1 - self.G_mesh[self.cond2]) / self.S_mesh[self.cond2]) - self.L**2 / self.Y_mesh[self.cond2])
            self.fee[self.cond3] = 0
        else:
            self.fee[self.cond1] = self.gamma / (1 - self.gamma) *\
                (self.L * np.sqrt((1 - self.gamma) * self.S_mesh[self.cond1]) - self.Y_mesh[self.cond1])
            self.fee[self.cond2] = self.gamma / (1 - self.gamma) *\
                (self.L * np.sqrt((1 - self.gamma) / self.S_mesh[self.cond2]) - self.L**2 / self.Y_mesh[self.cond2])
            self.fee[self.cond3] = 0
            
    def update_y_next(self):
        if self.dynamic_fee:
            self.y_next[self.cond1] = self.L * np.sqrt((1 - self.G_mesh[self.cond1]) * self.S_mesh[self.cond1])
            self.y_next[self.cond2] = self.L * np.sqrt(self.S_mesh[self.cond2] / (1 - self.G_mesh[self.cond2]))
            self.y_next[self.cond3] = self.Y_mesh[self.cond3]
        else:
            self.y_next[self.cond1] = self.L * np.sqrt((1 - self.gamma) * self.S_mesh[self.cond1])
            self.y_next[self.cond2] = self.L * np.sqrt(self.S_mesh[self.cond2] / (1 - self.gamma))
            self.y_next[self.cond3] = self.Y_mesh[self.cond3]
            
    def update_cont_value(self):
        if self.dynamic_fee:
            l_grid = np.abs(self.Y_grid[None, None, :] - self.y_next[..., None]).argmin(axis=-1)
            for j in range(self.nS):
                self.cont_value[j] = np.dot(self.V[:, l_grid[j]].transpose(1, 2, 0), self.transition_probs[j])
        else:
            l_grid = np.abs(self.Y_grid[None, None, :] - self.y_next[..., None]).argmin(axis=-1)
            for j in range(self.nS):
                self.cont_value[j] = np.dot(self.V[:, l_grid[j]].transpose(1, 0), self.transition_probs[j])

    def update_value_function(self):
        progress_bar = tqdm(total=self.T, desc="Dynamic Programming")
        for t in reversed(range(self.T)):
            self.find_condition()
            self.update_fee()
            self.update_y_next()
            self.update_cont_value()
            if self.dynamic_fee:
                exit_values = (self.L**2 * self.S_mesh + self.Y_mesh**2) / self.Y_mesh
                exit_values = exit_values[:, :, 0]
                total_val = self.fee + self.cont_value
                best_val = np.max(total_val, axis=2)
                best_gamma_idx = np.argmax(total_val, axis=2)
                self.best_gamma = self.gamma_grid[best_gamma_idx]
                self.V = np.maximum(exit_values, best_val)
            else:
                exit_values = (self.L**2 * self.S_mesh + self.Y_mesh**2) / self.Y_mesh
                self.V = np.maximum(exit_values, self.fee + self.cont_value)
            progress_bar.update(1)
        progress_bar.close()
        self.save_value_gamma()
        
    def save_value_gamma(self):
        V_df = pd.DataFrame(self.V)
        if self.dynamic_fee:
            V_df.to_csv(f'results/steps_{self.T}/V_dynamic_gamma.csv', index=False)
            best_gamma_df = pd.DataFrame(self.best_gamma)
            best_gamma_df.to_csv(f'results/steps_{self.T}/best_gamma.csv', index=False)
        else:
            V_df.to_csv(f'results/steps_{self.T}/V_fix_gamma.csv', index=False)
            
        PV0 = np.zeros((self.nS, self.nY))

        for j in range(self.nS):
            for k in range(self.nY):
                PV0[j, k] = (self.L**2 * self.S_grid[j] + self.Y_grid[k]**2) / self.Y_grid[k]

        PV0_df = pd.DataFrame(PV0)
        
        PV0_df.to_csv(f'results/steps_{self.T}/initial_pool_value.csv', index=False)
        

# %%
if __name__ == "__main__":
    import os
    
    # for T in [10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
    # for T in [100, 200, 500, 1000, 2000, 5000]:
    for T in [100]:
        os.makedirs(f'results/steps_{T}', exist_ok=True)
        dp = DPAMM(nS, nY, nG, T, L, gamma, sigma, dt,
                smax, smin, ymax, ymin, gmax, gmin)
        dp.update_value_function()
    
# %%