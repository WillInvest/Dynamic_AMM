# importing src directory
import sys
sys.path.append('..')
# experiment imports
import os
import math
import numpy as np
import random
from scipy.stats import truncnorm
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
# project imports
from amm.amm import AMM, SimpleFeeAMM
from amm.fee import TriangleFee, PercentFee, NoFee
# data imports
from data.kaiko import fetch_data
from api_key.my_api_key import api_key


def calibrate_gbm(T, N, dt, asset, start_date, end_date, freq, api_key):
    """
    calibrate geometric brownian motion
    
    mu (float): drift coefficient
    sigma (float): diffusion coefficient
    S0 (float): initial value
    T (float): terminal time
    N (int): number of time steps
    dt (float): time step size
    asset (str): token for aset (e.g. btc, eth)
    start_date (str): start date for data (YYYY-MM-DD)
    end_date (str): end date for data (YYYY-MM-DD)
    freq (str): frequency of data (1h, 1d, 1w)

    return numpy.ndarray: simulated gbm path
    """
    # if os.path.exists(f"/Users/andrewcarranti/CODE/SHIFT/2024/py_repo/post_refactor/AMM-Python/src/analyze/crypto_data/{asset}-usd_{start_date}_{end_date}_{freq}.csv"):
    #     data =  pd.read_csv(f"/Users/andrewcarranti/CODE/SHIFT/2024/py_repo/post_refactor/AMM-Python/src/analyze/crypto_data/{asset}-usd_{start_date}_{end_date}_{freq}.csv")["price"]
    if os.path.exists(f"/analyze/crypto_data/{asset}-usd_{start_date}_{end_date}_{freq}.csv"):
        data =  pd.read_csv(f"/analyze/crypto_data/{asset}-usd_{start_date}_{end_date}_{freq}.csv")["price"]
    else: data = fetch_data(api_key, asset, start_date, end_date, freq)

    # use data to calibrate gbm
    returns = np.log(data / data.shift(1)) # get returns
    mu = returns.mean() * 252  # annualized return
    sigma = returns.std() * 252 ** 0.5 # annualized volatility
    S0 = data.iloc[-1] # get LAST price in series
    # generate gbm path
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W 
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return S

def sim1(n, asset, start_dt, end_dt, frequency):
    """
    simulate AMM market with data calibrated GBM for external oracles and trading agents
    n (int): number of simulations
    pair (str): asset pair for data (e.g. btc-usd, eth-usd)
    start_dt (str): start date for data (YYYY-MM-DD)
    end_dt (str): end date for data (YYYY-MM-DD)
    frequency (str): frequency of data (1h, 1d, 1w)
    return list: list of dataframes for each simulation 
    """
    amm_sims = [] # create list to store dfs from each simulation of amms
    crypto_data = fetch_data(api_key, asset, start_dt, end_dt, frequency) # fetch data from kaiko - WON'T vary per simulation
    for set in range(n): # create new set of amms & run trades on each - do this n times
        nofeeAMM = SimpleFeeAMM(fee_structure = NoFee())
        percentAMM = SimpleFeeAMM(fee_structure = PercentFee(0.01))
        triAMM = SimpleFeeAMM(fee_structure = TriangleFee(0.003, 0.0001, -1)) # setup dfs to store simulations
        # create new set of dfs
        nofeeDF = pd.DataFrame(colummns=["AInv", "BInv", "LInv", "A", "B", "L", "FA", "FB", "FL"])
        percentDF = pd.DataFrame(colummns=["AInv", "BInv", "LInv", "A", "B", "L", "FA", "FB", "FL"])
        triDF = pd.DataFrame(colummns=["AInv", "BInv", "LInv", "A", "B", "L", "FA", "FB", "FL"])
        # store pairs of amm & df for updating
        amms = [(nofeeAMM, nofeeDF), (percentAMM, percentDF), (triAMM, triDF)]
        # create market df
        marketDF = pd.DataFrame(columns=["amm_xr", "real_xr", "tracking", "A20", "B20", "A50", "B50", "A200", "B200"])
        

        print(marketDF)

        # # iterate over each time step in crypto price path
        # for t, trade in enumerate(crypto_data):
            

        #     print("")

    # return amm_sims # return list of dfs for each simulation
        
        
        
        
        # for trade in range(10000): # run trades for each set of AMMs
        # # # ADD TRADE CALLS FROM AGENTS HERE # #
        #     asset_out, asset_in, asset_in_n = "A", "B", 1 # parse input
        # # # ADD TRADE CALLS FROM AGENTS HERE # #
        #     succ1, info1 = nofeeAMM.trade_swap(asset_out, asset_in, asset_in_n) # call trade for each AMM
        #     succ2, info2 = triAMM.trade_swap(asset_out, asset_in, asset_in_n) # (for each fee type)
        #     succ3, info3 = percentAMM.trade_swap(asset_out, asset_in, asset_in_n)
        #     #
        #     for amm, df in amms: # update dfs with each trade
        #         new_row = {'AInv': amm.portfolio['A'], 'BInv': amm.portfolio['B'], 'LInv': amm.portfolio['L'], 
        #                 'A': info1['asset_delta']['A'], 'B': info1['asset_delta']['B'], 'L': info1['asset_delta']['L'], 
        #                 'FA': amm.fees['A'], 'FB': amm.fees['B'], 'FL': amm.fees['L']}
        #         df = df.append(new_row, ignore_index=True)

        

        
        

if __name__ == "__main__":
    sim1(2, "btc-usd", '2023-02-01T00:00:00Z', '2024-03-01T00:00:00Z', "1d")







# # Parameters
# mu = 0 # Drift coefficient
# sigma = 0.25  # Diffusion coefficient
# S0 = btc[0]  # Initial value eth[0]
# T = 1.0  # Terminal time
# N = 1440  # Number of time steps
# dt = T / N  # Time step size

# # Generate GBM path
# gbm_path = geometric_brownian_motion(mu, sigma, S0, T, N, dt)