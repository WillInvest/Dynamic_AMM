import os
import sys
import socket
from collections import deque
import pandas as pd
# Get the path to the Dynamic_AMM directory
sys.path.append(f'{os.path.expanduser("~")}/Dynamic_AMM')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from env.new_amm import AMM
import math
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import norm
import yfinance as yf
import time
class MarketSimulator:
    def __init__(self,
                 start_price=500,
                 mu=0.06,
                 sigma=None,
                 dt=1/(252*6.5*60*60),  # 1 second in trading days
                 steps=23400,  # number of seconds in 20 trading days
                 spread=0.005,
                 clustering=False,
                 seed=None):
        
        # Initialize random number generator, only if seed is provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()  # Random behavior without a seed
        self.initial_price = start_price
        self.AP = start_price
        self.BP = start_price
        self.current_price = start_price
        self.mu = mu
        self.random_sigma = True if sigma<0 else False
        self.sigma_t = sigma if sigma>=0 else self.get_random_sigma()
        self.spread = spread
        self.dt = dt
        self.index = 0
        self.steps = steps
        self.clustering = clustering
        self.spy_mean = 0.0113 * np.sqrt(252)  # Annualized mean volatility
        self.spy_std = 0.0076 * np.sqrt(252)  # Annualized standard deviation of volatility
        # Initialize deque for GARCH variance calculation
        self.reset_sigma()

    def get_volatility_stats(self):
        # Download SPY data
        spy = yf.download('SPY', start='2020-01-01', end='2024-10-01')

        # Calculate log returns and rolling volatility (20-day window, adjust as needed)
        spy['Log_Returns'] = np.log(spy['Adj Close'] / spy['Adj Close'].shift(1))
        spy['Volatility'] = spy['Log_Returns'].rolling(window=20).std()
        spy = spy.dropna()

        # Extract the volatility values
        historical_volatility = spy['Volatility'].values

        # Calculate mean and standard deviation of the historical volatility
        mean = np.mean(historical_volatility)
        std = np.std(historical_volatility)
        print(f"SPY from 2020-01-01 to 2024-10-01: Mean Volatility: {mean:.4f}, Std Dev: {std:.4f}")

        return mean, std

    def get_random_sigma(self):
        # Sample a sigma value from the normal distribution based on historical mean and std
        # sampled_sigma = norm.rvs(loc=self.spy_mean, scale=self.spy_std, random_state=self.rng)
        sampled_sigma = self.rng.uniform(0.05, 0.35)
        # Ensure sampled sigma is positive (since volatilities can't be negative)
        return sampled_sigma

    def get_bid_price(self, token):
        if token == "A":
            return self.AP - self.spread
        elif token == "B":
            return self.BP - self.spread
        else:
            print("Invalid input")
            
    def get_ask_price(self, token):
        if token == "A":
            return self.AP + self.spread
        elif token == "B":
            return self.BP + self.spread
        else:
            print("Invalid input")

    def next(self):
        self.index += 1
        shock1 = self.rng.normal()
        shock2 = self.rng.normal()

        # change sigma is steps is a multiple of 1000
        if self.index % 60 == 0:
            if self.random_sigma:
                if self.clustering:
                    residual_shock = self.rng.normal()
                    # Generate next volatility using the GARCH(2,2) process
                    prev_variance = list(self.past_variance)
                    prev_residual = list(self.past_residuals)
                    new_variance = (
                        self.alpha0 +
                        self.alpha1 * prev_residual[-1]**2 +
                        self.alpha2 * prev_residual[-2]**2 +
                        self.beta1 * prev_variance[-1] +
                        self.beta2 * prev_variance[-2]
                                )
                    self.sigma_t = np.sqrt(new_variance)
                    self.past_variance.append(new_variance)
                    self.past_residuals.append(residual_shock*self.sigma_t)
                    
                else:
                    # Randomly choose a sigma within the range [0.05, 0.35]
                    self.sigma_t = self.get_random_sigma()

        # Update the current price using the GBM formula with the determined sigma
        self.AP *= np.exp(
            (self.mu - 0.5 * self.sigma_t ** 2) * self.dt + self.sigma_t * np.sqrt(self.dt) * shock1)
        self.BP *= np.exp(
            (self.mu - 0.5 * self.sigma_t ** 2) * self.dt + self.sigma_t * np.sqrt(self.dt) * shock2)
        
    def reset_sigma(self):
        self.past_variance = deque([0.2**2, 0.2**2], maxlen=2)  # Reset the GARCH variance with the initial value
        self.past_residuals = deque([0.2, 0.2], maxlen=2)  # Reset the residuals with the initial value
        # GARCH(2,2) parameters from the trained model
        self.alpha0 = 3.6495997020763665e-06
        self.alpha1 = 0.09999989631275566
        self.alpha2 = 0.09999987518264385
        self.beta1 = 0.39000083805952757
        self.beta2 = 0.38999949273198525
        
    def reset(self):
        self.AP = self.initial_price
        self.BP = self.initial_price
        self.index = 0
        self.reset_sigma()
        
        
 # Define a function to simulate the market and calculate the volatility       
        
def simulate_and_visualize(market, window_size=100):
    # Lists to store time, mid prices, and calculated volatility values for visualization
    time_steps = []
    mkt_mid = []
    sigma_values = []
    log_returns = []
    sum_log_returns = 0.0
    sum_log_returns_sq = 0.0

    # Run the simulation for the given number of steps
    for _ in range(market.steps):
        market.next()
        mkt_ask = market.get_ask_price('A')
        mkt_bid = market.get_bid_price('B')
        mid_price = (mkt_ask + mkt_bid) / 2

        # Calculate the log return
        if len(mkt_mid) > 0:
            last_price = mkt_mid[-1]
            log_return = np.log(mid_price / last_price)
            sum_log_returns += log_return
            sum_log_returns_sq += log_return ** 2
            log_returns.append(log_return)

            # Maintain the window size for log returns
            if len(log_returns) > window_size:
                oldest_return = log_returns.pop(0)
                sum_log_returns -= oldest_return
                sum_log_returns_sq -= oldest_return ** 2

            # Calculate volatility when the window size is met
            if len(log_returns) == window_size:
                mean_log_return = sum_log_returns / window_size
                variance = (sum_log_returns_sq / window_size) - (mean_log_return ** 2)
                current_sigma = np.sqrt(variance)
                sigma_values.append(current_sigma)
                time_steps.append(market.index)

        mkt_mid.append(mid_price)
        market.index += 1

    return time_steps, sigma_values, log_returns


# test whether the rule based Agent can take all arbitrage opportunities

def simulate_amm_with_market(amm, market):
    # Fetch reserves
    reserve_a, reserve_b = amm.get_reserves()
    amm_ask = (reserve_b / reserve_a) * (1 + amm.fee)
    amm_bid = (reserve_b / reserve_a) / (1 + amm.fee)
    
    market_ask = market.get_ask_price('A') / market.get_bid_price('B')
    market_bid = market.get_bid_price('A') / market.get_ask_price('B')
    
    if amm_ask < market_bid:
        swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid/(1+amm.fee))) / amm.reserve_a
    elif amm_bid > market_ask:
        swap_rate = math.sqrt((amm.reserve_a*amm.reserve_b*market_ask*(1+amm.fee)))/amm.reserve_b - 1
    else:
        swap_rate = 0
        
    amm.swap(swap_rate)
    reserve_a, reserve_b = amm.get_reserves()
    new_amm_ask = (reserve_b / reserve_a) * (1 + amm.fee)
    new_amm_bid = (reserve_b / reserve_a) / (1 + amm.fee)
    
    return amm_ask, amm_bid, new_amm_ask, new_amm_bid, swap_rate!=0
    
def calculate_autocorrelation_squared_returns(log_returns, max_lag=20):
    # Ensure squared_returns is 1-dimensional
    squared_returns = np.array(log_returns)**2
    squared_returns = squared_returns.flatten()  # Flatten to ensure it's 1-dimensional

    autocorrelations = [pd.Series(squared_returns).autocorr(lag) for lag in range(1, max_lag+1)]
    
    # Plotting the autocorrelation
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, max_lag+1), autocorrelations)
    plt.title('Autocorrelation of Squared Returns')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()
    
    return autocorrelations

def calculate_ljung_box_test(log_returns, lags=10):
    squared_returns = np.array(log_returns)**2
    lb_test = acorr_ljungbox(squared_returns, lags=[lags], return_df=True)
    p_value = lb_test['lb_pvalue'].iloc[0]
    return lb_test, p_value

if __name__ == '__main__':
 
    # Initialize two market simulators: one with clustering and one without
    market_with_clustering = MarketSimulator(clustering=True)
    market_without_clustering = MarketSimulator(clustering=False)

    # Simulate both markets
    time_with_clustering, sigma_with_clustering, log_returns_with_clustering = simulate_and_visualize(market_with_clustering)
    time_without_clustering, sigma_without_clustering, log_returns_without_clustering = simulate_and_visualize(market_without_clustering)

    # Calculate and display statistics for clustering effect
    autocorrelations_with = calculate_autocorrelation_squared_returns(log_returns_with_clustering)
    autocorrelations_without = calculate_autocorrelation_squared_returns(log_returns_without_clustering)
    
    # Perform Ljung-Box test
    lb_test_with, p_value_with = calculate_ljung_box_test(log_returns_with_clustering)
    lb_test_without, p_value_without = calculate_ljung_box_test(log_returns_without_clustering)

    def remove_outliers(data, time_data, factor=3):
        # Calculate the first and third quartile (Q1 and Q3)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1  # Interquartile range
    
        # Calculate the lower and upper bounds to identify outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
    
        # Filter the data to only include values within the bounds
        filtered_data = [(x, t) for x, t in zip(data, time_data) if lower_bound <= x <= upper_bound]
    
        # Separate the filtered values back into two lists
        filtered_values, filtered_time = zip(*filtered_data)
        return list(filtered_time), list(filtered_values)

    # Filter sigma values and time to remove outliers
    filtered_time_with_clustering, filtered_sigma_with_clustering = remove_outliers(sigma_with_clustering, time_with_clustering)
    filtered_time_without_clustering, filtered_sigma_without_clustering = remove_outliers(sigma_without_clustering, time_without_clustering)

    # Plot the results
    plt.figure(figsize=(14, 6))

    # Plot with clustering
    plt.subplot(1, 2, 1)
    plt.plot(filtered_time_with_clustering, filtered_sigma_with_clustering, color='blue', label='Autocorrelation: {:.4f}\nLjung-Box p-value: {:.4f}'.format(
        autocorrelations_with[0], p_value_with))
    plt.title('Volatility with Clustering (GARCH)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Volatility (sigma)')
    plt.grid(True)
    plt.legend()

    # Get the y-axis limits from the filtered plot with clustering
    y_limits = plt.gca().get_ylim()

    # Plot without clustering using the same y-axis limits
    plt.subplot(1, 2, 2)
    plt.plot(filtered_time_without_clustering, filtered_sigma_without_clustering, color='red', label='Autocorrelation: {:.4f}\nLjung-Box p-value: {:.4f}'.format(
        autocorrelations_without[0], p_value_without))
    plt.title('Volatility without Clustering')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Volatility (sigma)')
    plt.ylim(y_limits)  # Set y-axis limits to match the "with clustering" plot
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('volatility_comparison.png')
    plt.show()