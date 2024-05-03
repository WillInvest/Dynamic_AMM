# importing src directory
import sys
sys.path.append('..')
# experiment imports
import os
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import minimize
# project imports
# data imports
from data.kaiko import fetch_data



def geometric_brownian_motion(mu, sigma, S0, T, N, dt):
    """
    Generate geometric brownian motion.

    Parameters:
        mu (float): Drift coefficient.
        sigma (float): Diffusion coefficient.
        S0 (float): Initial value.
        T (float): Terminal time.
        N (int): Number of time steps.
        dt (float): Time step size.

    Returns:
        numpy.ndarray: Simulated GBM path.
    """
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
    X = (mu - 0.5 * sigma*2)**t + sigma * W 
    S = S0 * np.exp(X)  # Geometric Brownian motion
    return S


# # NOTE TODO: BELOW FUNCTIONS CURRENTLY NOT IN USE # #

def gbm_assumption_test(log_returns, show_all_results=False):
    adf_result = adfuller(log_returns) # check for stationarity
    shapiro_result = shapiro(log_returns) # check for normality
    lb_result = acorr_ljungbox(log_returns, lags=[10], return_df=True) # check for independence (autocorrelation)
    if show_all_results:
        # stationarity, normality, independence
        print("ADF Statistic:", adf_result[0]) # check for stationarity
        print("P-value:", adf_result[1])
        print("Critical Values:", adf_result[4])
        print("Stationary:", adf_result[1] <= 0.05)
        print("Shapiro-Wilk Test Statistic:", shapiro_result[0])
        print("p-value:", shapiro_result[1])
        print("Normal:", shapiro_result[1] > 0.05)
        print("Ljung-Box test:")
        print(lb_result)
        print("Independent:", lb_result['lb_pvalue'].iloc[0] > 0.05)
        print("-"*50)
    else:
        print("Stationary:", adf_result[1] <= 0.05)
        print("Normal:", shapiro_result[1] > 0.05)
        print("Independent:", lb_result['lb_pvalue'].iloc[0] > 0.05)
        print("-"*50)
    # NOTE: code to iteratively difference until stationary - not needed for now
    # if adf_result[1] > 0.05:  # if not stationary, iteratively difference until achieved
    #     for d in range(1, max_lag + 1):
    #         diff_data = diff(log_returns, k_diff=d)
    #         adf_result = adfuller(diff_data)
    #         print(f"ADF result after differencing level {d}: {adf_result[0]}, p-value: {adf_result[1]}")
    #         if adf_result[1] <= 0.05:
    #             print("Achieved stationarity with differencing level:", d)
    #             diff_data = diff_data
    #             break


# define the negative log likelihood function
def neg_log_likelihood(params, log_returns):
    """
    calculate negative log likelihood of a normal distribution for calibrating GBM
    params: tuple, mu and sigma
    """
    mu, sigma = params # define mu and sigma
    estimated_mu = np.mean(log_returns) # estimate mu
    estimated_var = np.sum((log_returns - estimated_mu)**2) / len(log_returns) # estimate variance
    return 0.5 * len(log_returns) * np.log(2 * np.pi * estimated_var) + 0.5 / estimated_var * np.sum((log_returns - mu)**2) # return negative log likelihood

def calibrate_gbm(asset, data, frequency, T, N, type, show_all_results=False):
    """
    calibrate geometric brownian motion for next period (t=0 is last observation in data)
    
    calibrate gbm model by pulling data from kaiko api

    asset (str): asset to calibrate
    data (pd.DataFrame): price data w/ column 'price'
    freq (str): frequency of data (1h, 1d, 1w)
    T (float): terminal time
    N (int): number of time steps
    type (str): type of calibration (reg, mle)
    max_lag (int): maximum lag for autocorrelation test (default=10)
    alpha (float): significance level for hypothesis tests (default=0.05)

    return mu (float), sigma (float), S (numpy.ndarray)
    """

    if type == "reg":
        returns = np.log((data / data.shift(1)).dropna()) # get returns
        gbm_assumption_test(returns) # test gbm assumptions
        mu = returns.mean() * 365.25  # annualized return
        sigma = returns.std() * 365.25 ** 0.5 # annualized volatility
        if show_all_results:
            print(f'Estimated {asset} {frequency} Mu:', round(mu, 2), 'Estimated Annualized Mu:', round(mu * 365.25,2))
            print(f'Estimated {asset} {frequency} Sigma:', round(sigma, 2), 'Estimated Annualized Sigma:', round(sigma * 365.25**0.5, 2))
        S0 = data.iloc[0] # NOTE get FIRST price in series 
        dt = T / N # time step size
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  # Standard Brownian motion
        X = (mu - 0.5 * sigma**2) * t + sigma * W 
        S = S0 * np.exp(X)  # Geometric Brownian motion   
        return mu, sigma, S
    
    elif type == "mle":
        log_returns = np.log(1 + data.pct_change().dropna()) # calculate log returns
        result = minimize(neg_log_likelihood, [0.05, 0.2], args=(log_returns,), bounds=((None, None), (1e-4, None))) # minimize the negative log-likelihood
        mu = result.x[0] * 365.25 # annualize mu
        sigma = result.x[1] * 365.25**0.5 # annualize sigma
        if show_all_results:
            print(f'Estimated {asset} {frequency} Mu:', round(result.x[0],5), 'Estimated Annualized Mu:', round(mu, 5)) # using 365.25 instead of 252 bcs operate 24/7
            print(f'Estimated {asset} {frequency} Sigma:', round(result.x[1],5), 'Estimated Annualized Sigma:', round(sigma, 5))
        S0 = data.iloc[0] # get FIRST price in series
        dt = T / N # time step size
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  # standard BM
        X = (mu - 0.5 * sigma**2) * t + sigma * W 
        S = S0 * np.exp(X)  # GBM
        return mu, sigma, S
    

def get_price_data(pair, start_date, end_date, freq, api_key):
    """
    get price data from kaiko api or local storage

    asset (str): asset symbol
    start_date (str): start date of data
    end_date (str): end date of data
    freq (str): frequency of data (1h, 1d, 1w)
    api_key (str): kaiko api key

    return pd.DataFrame: price data
    """
    asset1 = pair.split("-")[0] 
    asset2 = pair.split("-")[1]
    # check if data exists, if not fetch data
    if os.path.exists(f"/data/crypto_data/{asset1}_{start_date}_{end_date}_{freq}.csv"):
        data1 =  pd.read_csv(f"/data/crypto_data/{asset1}_{start_date}_{end_date}_{freq}.csv")["price"]
    else: data1 = fetch_data(api_key, asset1, start_date, end_date, freq)
    # convert timestamp to datetime & price to numeric for asset 1 (A)
    data1['timestamp'] = pd.to_datetime(data1['timestamp'], unit='ms')
    data1[f'{asset1}_mrkt_price'] = pd.to_numeric(data1[f'{asset1}_mrkt_price'])
    # check if data exists, if not fetch data
    if os.path.exists(f"/data/crypto_data/{asset2}_{start_date}_{end_date}_{freq}.csv"):
        data2 =  pd.read_csv(f"/data/crypto_data/{asset2}_{start_date}_{end_date}_{freq}.csv")["price"]
    else: data2 = fetch_data(api_key, asset2, start_date, end_date, freq)
    # convert timestamp to datetime & price to numeric for asset 2 (B)
    data2['timestamp'] = pd.to_datetime(data2['timestamp'], unit='ms')
    data2[f'{asset2}_mrkt_price'] = pd.to_numeric(data2[f'{asset2}_mrkt_price'])
    # merge dataframes on timestamp saving price for each asset denominated in USD for storing AMM market data
    return pd.merge(data1, data2, on='timestamp', how='inner')


def main():
    return None

if __name__ == '__main__':
    main()