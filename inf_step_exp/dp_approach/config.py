# Configuration file for DP approach parameters
# This file contains all the parameters used in dp.py and dp_results.ipynb
import numpy as np

# Grid parameters
nS = 200  # Number of price states
nY = 100  # Number of Y reserve states
nG = 500  # Number of gamma states (for dynamic fee)

# Time parameters
T = 100   # Number of time steps
dt = 1/365  # Time step size (daily)

# AMM parameters
L = 1.0   # Liquidity parameter
# gamma = 0.003 # Fixed fee rate (None for dynamic fee)
gamma = 0.003
sigma = 0.2   # Volatility parameter

# Grid bounds
smax = 1.5   # Maximum price
smin = 0.5   # Minimum price
ymax = 1.5   # Maximum Y reserve
ymin = 0.5   # Minimum Y reserve
gmax = 0.9995  # Maximum gamma (fee rate)
gmin = 0.0005  # Minimum gamma (fee rate)

S_grid = np.linspace(smin, smax, nS)
Y_grid = np.linspace(ymin, ymax, nY)
gamma_grid = np.linspace(gmin, gmax, nG)

# Price dynamics
price_dynamics = 'GBM'  # Geometric Brownian Motion
