# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from memory_profiler import profile


# Compute GBM transition matrix
def transition_prob(s0, s1, sigma, dt):
    return (1 / (s1 * sigma * np.sqrt(2 * np.pi * dt))) * \
        np.exp(- (np.log(s1) - np.log(s0))**2 / (2 * sigma**2 * dt))

# Parameters
T = 100000         # number of time steps
nS = 50          # number of price grid points
nY = 50          # number of y reserve levels
nG = 50          # number of gamma grid points
L = 1.0          # AMM invariant constant
gamma = 0.01     # arbitrage threshold
sigma = 0.2      # annual volatility
dt = 1/252       # time step size in years (daily)

# Grids
S_grid = np.linspace(0.5, 1.5, nS)
Y_grid = np.linspace(0.5, 1.5, nY)
gamma_grid = np.linspace(0.0005, 0.01, nG)
S_mesh, Y_mesh = np.meshgrid(S_grid, Y_grid, indexing='ij')
# Step 1: Create 3D meshgrid (nS, nY, nG)
S3, Y3, G3 = np.meshgrid(S_grid, Y_grid, gamma_grid, indexing='ij')  # All shape (nS, nY, nG)
# Initialize fee and y_next arrays
fee3 = np.zeros_like(S3)
y_next3 = np.zeros_like(Y3)
continuation3 = np.zeros_like(S3)

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
print(f"VV.shape: {VV.shape}")
progress_bar = tqdm(total=T, desc="Dynamic Programming")

# Instead of nested loops, vectorize the S and Y operations
for t in reversed(range(T)):
    continuation3.fill(0)
    fee3.fill(0)
    y_next3.fill(0)

    # Step 2: Compute pool price and exit values
    P3 = Y3**2 / L**2
    
    # Three conditions
    cond1 = S3 > P3 / (1 - G3)
    cond2 = S3 < P3 * (1 - G3)
    cond3 = ~(cond1 | cond2)

    # Compute fee and y_next under each condition
    fee3[cond1] = G3[cond1] / (1 - G3[cond1]) * (L * np.sqrt((1 - G3[cond1]) * S3[cond1]) - Y3[cond1])
    y_next3[cond1] = L * np.sqrt((1 - G3[cond1]) * S3[cond1])

    fee3[cond2] = G3[cond2] / (1 - G3[cond2]) * (L * np.sqrt((1 - G3[cond2]) / S3[cond2]) - L**2 / Y3[cond2])
    y_next3[cond2] = L * np.sqrt(S3[cond2] / (1 - G3[cond2]))
    
    y_next3[cond3] = Y3[cond3]
    
    l_grid = np.abs(Y_grid[None, None, :] - y_next3[..., None]).argmin(axis=-1)
    
    for j in range(nS):
        continuation3[j] = np.dot(VV[:, l_grid[j]].transpose(1, 2, 0), transition_probs[j])

    # Step 5: Total value and argmax
    total_val = fee3 + continuation3  # shape (nS, nY, nG)
    best_val = np.max(total_val, axis=2)  # shape (nS, nY)
    best_gamma_idx = np.argmax(total_val, axis=2)
    best_gamma = gamma_grid[best_gamma_idx]

    # Step 6: Choose between exit and continuation
    exit_vals = (L**2 * S_mesh + Y_mesh**2) / Y_mesh  # shape (nS, nY)
    VV = np.maximum(exit_vals, best_val)

    progress_bar.update(1)

progress_bar.close()

V_df = pd.DataFrame(VV)
V_df.to_csv('Dynamic_fee_value.csv')
best_gamma_df = pd.DataFrame(best_gamma)
best_gamma_df.to_csv('best_gamma.csv')

# convert to dataframe

best_gamma_df.index = [f"S={S_grid[j]:.2f}" for j in range(nS)]
best_gamma_df.columns = [f"Y={Y_grid[k]:.2f}" for k in range(nY)]

plt.figure(figsize=(12, 10))
sns.heatmap(best_gamma_df, cmap='YlOrRd', fmt='.2f', 
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Best Gamma Heatmap')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()

# %%

# convert to dataframe

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

v_fix_gamma_df = pd.read_csv('V_fix_gamma.csv', index_col=0)

plt.figure(figsize=(12, 10))
sns.heatmap(v_fix_gamma_df, cmap='YlOrRd', fmt='.2f',
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Value Function V[0] - PV0 Heatmap')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()

# %%

# plot diff between V_df and v_fix_gamma_df

plt.figure(figsize=(12, 10))
sns.heatmap(V_df - v_fix_gamma_df, cmap='YlOrRd', fmt='.2f',
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Value Function V[0] - PV0 Heatmap')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()

# %%


# %%

dff = pd.DataFrame(V_df.to_numpy() - v_fix_gamma_df.to_numpy())
dff.index = [f"{S_grid[j]:.2f}" for j in range(nS)]
dff.columns = [f"{Y_grid[k]:.2f}" for k in range(nY)]

plt.figure(figsize=(12, 10))
sns.heatmap(dff, cmap='YlOrRd', fmt='.2f',
            xticklabels=5, yticklabels=5)  # Show every 5th label to avoid overcrowding
plt.title('Dynamic Fee - Fix Fee')
plt.xlabel('Y Reserve')
plt.ylabel('Price (S)')
plt.gca().invert_yaxis()  # Invert y-axis
plt.tight_layout()
plt.show()






# %%
