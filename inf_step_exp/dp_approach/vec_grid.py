# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



# Compute GBM transition matrix
def transition_prob(s0, s1, sigma, dt):
    return (1 / (s1 * sigma * np.sqrt(2 * np.pi * dt))) * \
        np.exp(- (np.log(s1) - np.log(s0))**2 / (2 * sigma**2 * dt))

# Helper functions
def find_closest_index(grid, val):
    return np.argmin(np.abs(grid - val))

# Parameters
T = 10000         # number of time steps
nS = 200          # number of price grid points
nY = 200          # number of y reserve levels
# nG = 50          # number of gamma grid points
L = 1.0          # AMM invariant constant
gamma = 0.01     # arbitrage threshold
sigma = 0.2      # annual volatility
dt = 1/252       # time step size in years (daily)

# Grids
S_grid = np.linspace(0.5, 1.5, nS)
Y_grid = np.linspace(0.5, 1.5, nY)
# gamma_grid = np.linspace(0.0001, 0.05, nG)
S_mesh, Y_mesh = np.meshgrid(S_grid, Y_grid, indexing='ij')

fees = np.zeros_like(S_mesh)
y_next = np.zeros_like(Y_mesh)
continuation3 = np.zeros_like(S_mesh)
transition_probs = np.zeros((nS, nS))
for i in range(nS):
    for j in range(nS):
        transition_probs[i, j] = transition_prob(S_grid[i], S_grid[j], sigma, dt)
    transition_probs[i] /= transition_probs[i].sum()  # Normalize

# Initialize value function
VV = np.zeros((nS, nY))
for j in range(nS):
    for k in range(nY):
        VV[j, k] = (L**2 * S_grid[j] + Y_grid[k]**2) / Y_grid[k]
progress_bar = tqdm(total=T, desc="Dynamic Programming")

# Instead of nested loops, vectorize the S and Y operations
for t in reversed(range(T)):
    fees.fill(0)
    y_next.fill(0)
    continuation3.fill(0)
    
    # Compute all pool prices at once
    P_mesh = Y_mesh**2 / L**2
    
    # Vectorized fee calculations
    condition1 = S_mesh > P_mesh / (1-gamma)
    condition2 = S_mesh < P_mesh * (1-gamma)
    
    # Condition 1: S > P/(1-gamma)
    fees[condition1] = gamma/(1-gamma) * (L * np.sqrt((1-gamma) * S_mesh[condition1]) - Y_mesh[condition1])
    y_next[condition1] = L * np.sqrt((1-gamma) * S_mesh[condition1])
    
    # Condition 2: S < P*(1-gamma)  
    fees[condition2] = gamma/(1-gamma) * (L * np.sqrt((1-gamma) / S_mesh[condition2]) - L**2 / Y_mesh[condition2])
    y_next[condition2] = L * np.sqrt(S_mesh[condition2] / (1-gamma))
    
    # Condition 3: else
    condition3 = ~(condition1 | condition2)
    y_next[condition3] = Y_mesh[condition3]
    
    l_grid = np.abs(Y_grid[None, None, :] - y_next[..., None]).argmin(axis=-1)

    for j in range(nS):
        continuation3[j] = np.dot(VV[:, l_grid[j]].transpose(1, 0), transition_probs[j])
        
    # Compute all exit values at once
    exit_vals = (L**2 * S_mesh + Y_mesh**2) / Y_mesh
    VV = np.maximum(exit_vals, fees + continuation3)
    
    # Vectorized expected value computation
    # for j in range(nS):
    #     for k in range(nY):
    #         l = find_closest_index(Y_grid, y_next[j, k])
    #         weights = transition_probs[j]  # Pre-compute this outside the loop
    #         expected_val = np.dot(weights, VV[:, l])
    #         VV[j, k] = max(exit_vals[j, k], fees[j, k] + expected_val)
    progress_bar.update(1)

progress_bar.close()

# %%

# convert to dataframe
V_df = pd.DataFrame(VV)
V_df.to_csv('V_fix_gamma.csv')
V_df.index = [f"S={S_grid[j]:.2f}" for j in range(nS)]
V_df.columns = [f"Y={Y_grid[k]:.2f}" for k in range(nY)]

plt.figure(figsize=(12, 10))
sns.heatmap(V_df, cmap='YlOrRd', fmt='.2f', 
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Value Function V[0] Heatmap')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()

# %%

PV0 = np.zeros((1, nS, nY))

for j in range(nS):
    for k in range(nY):
        PV0[0, j, k] = (L**2 * S_grid[j] + Y_grid[k]**2) / Y_grid[k]

PV0_df = pd.DataFrame(PV0[0])
PV0_df.index = [f"S={S_grid[j]:.2f}" for j in range(nS)]
PV0_df.columns = [f"Y={Y_grid[k]:.2f}" for k in range(nY)]

plt.figure(figsize=(12, 10))
sns.heatmap(PV0_df, cmap='YlOrRd', fmt='.2f', 
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Value Function PV0 Heatmap')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()

# %%

plt.figure(figsize=(12, 10))
sns.heatmap(V_df - PV0_df, cmap='YlOrRd', fmt='.2f',
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Value Function V[0] - PV0 Heatmap')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()

# %%

# %%