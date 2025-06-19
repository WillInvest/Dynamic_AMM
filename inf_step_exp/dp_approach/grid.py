# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
T = 2000         # number of time steps
nS = 200          # number of price grid points
nY = 200          # number of y reserve levels
L = 1.0          # AMM invariant constant
gamma = 0.01     # arbitrage threshold
sigma = 0.2      # annual volatility
dt = 1/252       # time step size in years (daily)

# Grids
S_grid = np.linspace(0.5, 1.5, nS)
Y_grid = np.linspace(0.5, 1.5, nY)

# Compute GBM transition matrix
def transition_prob(s0, s1, sigma, dt):
    return (1 / (s1 * sigma * np.sqrt(2 * np.pi * dt))) * \
        np.exp(- (np.log(s1) - np.log(s0))**2 / (2 * sigma**2 * dt))

# Initialize value function
V = np.zeros((T+1, nS, nY))
for j in range(nS):
    for k in range(nY):
        V[T, j, k] = (L**2 * S_grid[j] + Y_grid[k]**2) / Y_grid[k]

# Helper functions
def find_closest_index(grid, val):
    return np.argmin(np.abs(grid - val))

total_progress = T * nS * nY
progress_bar = tqdm(total=total_progress, desc="Dynamic Programming")

# Dynamic Programming
for t in reversed(range(T)):
    for j in range(nS):
        for k in range(nY):
            S = S_grid[j]
            y = Y_grid[k]
            P = y**2 / L**2
            exit_val = (L**2 * S + y**2) / y

            if S > P / (1-gamma):
                fee = gamma/(1-gamma) * (L * np.sqrt((1-gamma) * S) - y)
                y_next = L * np.sqrt((1-gamma) * S)
            elif S < P * (1-gamma):
                fee = gamma/(1-gamma) * (L * np.sqrt((1-gamma) / S) - L**2 / y)
                y_next = L * np.sqrt(S / (1-gamma))
            else:
                fee = 0
                y_next = y

            l = find_closest_index(Y_grid, y_next)
            weights = np.array([transition_prob(S, S_grid[m], sigma, dt) for m in range(nS)])
            weights /= weights.sum()
            expected_val = np.dot(weights, V[t+1, :, l])

            V[t, j, k] = max(exit_val, fee + expected_val)
            progress_bar.update(1)

progress_bar.close()

# %%

# convert to dataframe
V_df = pd.DataFrame(V[0])
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
