# importing src directory
import sys
import os
sys.path.append('..')
# experiment imports
import math
import random
from datetime import datetime as dt
# project imports
from amm.amm import SimpleFeeAMM
from amm.fee import TriangleFee, PercentFee, NoFee
# data imports
from gbm.gbm import get_price_data, calibrate_gbm
from api_key.my_api_key import api_key

def sim1(n, pair, start_dt, end_dt, frequency, L0=1000000.0, spread=0.5):
    """
    simulate AMM market with data calibrated GBM for external oracles and trading agents
    n (int): number of simulations
    pair (str): asset pair for data (e.g. btc-eth)
    asset1_n (int): number of asset1 tokens
    asset2_n (int): number of asset2 tokens
    start_dt (str): start date for data (YYYY-MM-DD)
    end_dt (str): end date for data (YYYY-MM-DD)
    frequency (str): frequency of data (1h, 1d, 1w)
    L0 (int): number of initial LP tokens
    spread (float): spread for arbitrage agents (e.g. 0.5%)
    return list: list of dataframes for each simulation 
    """

    # # SIM STORAGE # #
    # create list to store dfs from each simulation of amms
    sim_amm_dfs= []
    # parse asset1 and asset2, create USD denominated pairs
    asset1 = pair.split("-")[0] 
    asset2 = pair.split("-")[1]
    # # DATA & GBM CALIBRATION # #
    # calculate time difference in years
    difference = dt.strptime(end_dt, '%Y-%m-%dT%H:%M:%SZ') - dt.strptime(start_dt, '%Y-%m-%dT%H:%M:%SZ')
    # using 365.25 to account for leap years
    T_years = difference.days / 365.25 
    # get data for gbm simulations assets
    price_df =  get_price_data(pair, start_dt, end_dt, frequency, api_key) # get data for assets
    # define new columns for amm & market data
    new_cols = [f'{asset1}_gbm_price', f'{asset2}_gbm_price', # gbm prices  #  NOTE: ADD WHEN LP MARKET -> f'L_gbm_price' # inventory of each asset
                f'{asset1}_inv', f'{asset2}_inv', 'L_inv', # AMM inventory of each asset
                f'F{asset1}_inv', f'F{asset2}_inv', 'FL_inv', # AMM inventory of fees for each asset
                f'{asset1}_dt', f'{asset2}_dt', 'L_dt', # AMM inventory changes for each asset
                f'F{asset1}_dt', f'F{asset2}_dt', 'FL_dt', # AMM inventory fee changes for each asset
                f'{asset1}_gbm_dt', f'{asset2}_gbm_dt', # gbm price changes
                f'{asset1}_mrkt_dt', f'{asset2}_mrkt_dt'] # market prices
    # assign new columns to gbm df with None values
    price_df = price_df.assign(**{col: None for col in new_cols})
    # # AMM PORTFOLIO @ t = 0 # #
    # evenly distribute assets
    amm_portfolio = {asset1: math.sqrt(L0), asset2: math.sqrt(L0), 'L': L0} # initial portfolio
    # add initial inventory to df
    price_df.at[0, f'{asset1}_inv'] = amm_portfolio[asset1] # add initial inventory to df
    price_df.at[0, f'{asset2}_inv'] = amm_portfolio[asset2] # add initial inventory to df
    price_df.at[0, 'L_inv'] = amm_portfolio['L'] # add initial inventory to df
    # # MARKET PORTFOLIO @ t = 0 # #
    market_init_portfolio = {"A": price_df.iloc[0][f'{asset1}_mrkt_price'], "B": price_df.iloc[0][f'{asset2}_mrkt_price'], "L": 0.0} # initial portfolio 
    # # TIME SERIES SIMULATIONS # #
    n_timesteps = len(price_df) # number of timesteps in data
    # for each simulation create new set of amms & run new set of trades
    for simulation in range(n):
        # # GBM CALIBRATION FOR ASSET 1 & 2 # #   (regular doesnt work right now)
        _,_,price_df[f'{asset1}_gbm_price'] = calibrate_gbm(asset1, price_df[f"{asset1}_mrkt_price"], frequency, T_years, n_timesteps, "mle") # calibrate gbm for asset1 w/ MLE
        _,_,price_df[f'{asset2}_gbm_price'] = calibrate_gbm(asset2, price_df[f"{asset2}_mrkt_price"], frequency, T_years, n_timesteps, "mle") # calibrate gbm for asset2 w/ MLE
        # # AMM CREATION # #
        nofeeAMM = SimpleFeeAMM(fee_structure = NoFee(), initial_portfolio=amm_portfolio)
        percentAMM = SimpleFeeAMM(fee_structure = PercentFee(0.01), initial_portfolio=amm_portfolio)
        triAMM = SimpleFeeAMM(fee_structure = TriangleFee(0.003, 0.0001, -1), initial_portfolio=amm_portfolio) 
        # store amms for iteration
        amms = [nofeeAMM, percentAMM, triAMM] # store amms
# BUG BELOW
        # # SIMULATION # #
        # iterate over each timestep in crypto market data
        for t in range(n_timesteps+1): 
            # set market data for each amm at initial timestep
            if t == 0:
                for amm in amms: 
                    amm.market_data = price_df
                    # set initial dt's to 0
                    dt_columns = [
                        f'{asset1}_gbm_dt', f'{asset2}_gbm_dt', f'{asset1}_mrkt_dt', f'{asset2}_mrkt_dt', 
                        f'{asset1}_dt', f'{asset2}_dt', 'L_dt', f'F{asset1}_dt', f'F{asset2}_dt', 'FL_dt']
                    for column in dt_columns: amm.market_data.at[t, column] = 0.0

            # update market data df with simulated transactions
            for amm in amms:
                # get ratio of asset1 to asset2
                amm_ratio = amm.market_data[f'{asset1}_inv'][t] / amm.market_data[f'{asset2}_inv'][t]
                gbm_ratio = amm.market_data[f'{asset1}_gbm_price'][t] / amm.market_data[f'{asset2}_gbm_price'][t]
                # # ARBITRAGE AGENT # #
                # if amm_ratio is greater than gbm_ratio by 0.5% or more 
                if amm_ratio > gbm_ratio * 1.005: # sell asset1, buy asset2
                    asset_out, asset_in, asset_in_n = asset1, asset2, random.choice(list(range(1, 50))) # modeling market efficiency
                elif amm_ratio * 1.005 < gbm_ratio: # sell asset2, buy asset1
                    asset_out, asset_in, asset_in_n = asset2, asset1, random.choice(list(range(1, 50))) # modeling market efficiency
                else: continue
                # call trade (on each AMM)
                succ, info = amm.trade_swap(asset_out, asset_in, asset_in_n)
                # # UPDATE DATA # #
                if succ: # if trade successful
                    # update inventory
                    # Define a dictionary to map DataFrame columns to their respective data sources
                    inventory_update = {f'{asset_out}_inv': amm.portfolio[asset_out], f'{asset_in}_inv': amm.portfolio[asset_in],
                        f'{asset_out}_dt': info['asset_delta'][asset_out], f'{asset_in}_dt': info['asset_delta'][asset_in],
                        f'F{asset_out}_inv': amm.fee_portfolio[asset_out], f'F{asset_in}_inv': amm.fee_portfolio[asset_in],
                        f'F{asset_out}_dt': info['fee_delta'][asset_out], f'F{asset_in}_dt': info['fee_delta'][asset_in],
                        f'{asset1}_gbm_dt': amm.market_data.at[t, f'{asset1}_gbm_price'] - amm.market_data.at[t-1, f'{asset1}_gbm_price'],
                        f'{asset2}_gbm_dt': amm.market_data.at[t, f'{asset2}_gbm_price'] - amm.market_data.at[t-1, f'{asset2}_gbm_price'],
                        f'{asset1}_mrkt_dt': amm.market_data.at[t, f'{asset1}_mrkt_price'] - amm.market_data.at[t-1, f'{asset1}_mrkt_price'],
                        f'{asset2}_mrkt_dt': amm.market_data.at[t, f'{asset2}_mrkt_price'] - amm.market_data.at[t-1, f'{asset2}_mrkt_price']}
                    # iterate over the dictionary to update the df
                    for key, value in inventory_update.items(): amm.market_data.at[t, key] = value
                # repeat for each amm here
            # repeats for each timestep here
        # # STORE DATA # #
        for amm in amms:
            # store amm data in list
            sim_amm_dfs.append(amm.market_data)
    # return list of simulation dfs
    return sim_amm_dfs 


def main():
    sim1(2, "btc-eth", '2023-02-01T00:00:00Z', '2024-02-05T00:00:00Z', "1d")


if __name__ == '__main__':
    main()







# # NOTES FROM LAST MEETING:
# FOCUS MORE ON TESTING FEES THROUGH SIM

# # EXPERIMENTS TODO: # #
# [1] run for large simulations and evaluate over time - explore different time periods to test from (different market conditions and lengths of historical windows) and different frequencies (1h, 1d, 1w)
# [2] identify GBM paths that deplete pools (depletion of liquidity) and have both fall in value (impermanent loss) to show how fee accumulation compares ot general trend (law of large #s)
        # impermanent loss evaluation could allow for an expected value calculation for LP returns (expected value of fees vs. impermanent loss)
# [3] use stock data to see how compares
# [4] make sure to highlight how different fee AMMs (basically fees) are affected by different market conditions and therefore how fee accumulation is affected

# # UPDATES # #
# [1] *importing stock data to use instead of crypto (more in line with goal application and can properly use GBM to simulate)
# [2] considering train/test split for calibrating GBM and simulating trades source data (not overly urgent given not forecasting)
# [3] maybe also considering changing source data from vwap if stick with crypto data
        # multiple price streams for multiple external oracles


