from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO, TD3
from tqdm import tqdm
import math
from collections import deque
import os
import pandas as pd
from datetime import datetime
from scipy import optimize as opt


def simulate_with_constant_fee_rate(fee_rate, sigma, config) -> dict:
    """
    urgent level determines whether the agent places an order or not.
    market competence determines how much percent of arbitrage opportunities will be taken by other traders in the market.
    """
    amm = AMM(fee=fee_rate)
    market = MarketSimulator(sigma=sigma,
                             mu=config['mu'],
                             spread=config['spread'],
                             dt=config['dt'],
                             start_price=config['start_price'],
                             steps=config['steps'])

    # price_distance = 0
    total_pnl = 0
    total_fee = 0
    total_volume = 0
    total_transactions = 0
    # mkt_mid = deque(maxlen=1000)
    # log_returns = deque(maxlen=999)    
    # sum_log_returns = 0
    # sum_log_returns_sq = 0
    # window_size = 999
    # sigma_values = []

    # Loop over market steps
    for _ in range(int(market.steps)):
        
        # mkt_ask = market.get_ask_price('A')
        # mkt_bid = market.get_bid_price('B')
        # mid_price = (mkt_ask + mkt_bid) / 2

        # if len(mkt_mid) > 0:
        #     last_price = mkt_mid[-1]
        #     log_return = np.log(mid_price / last_price)
        #     sum_log_returns += log_return
        #     sum_log_returns_sq += log_return ** 2
        #     log_returns.append(log_return)

        #     if len(log_returns) == window_size:
        #         oldest_return = log_returns[0]
        #         sum_log_returns -= oldest_return
        #         sum_log_returns_sq -= oldest_return ** 2

        # mkt_mid.append(mid_price)

        # if len(mkt_mid) == 1000:
        #     mean_log_return = sum_log_returns / window_size
        #     variance = (sum_log_returns_sq / window_size) - (mean_log_return ** 2)
        #     current_sigma = np.sqrt(variance)
        #     sigma_values.append(current_sigma)
            
            
        # Get trader observations
        market_ask = market.get_ask_price('A') / market.get_bid_price('B')
        market_bid = market.get_bid_price('A') / market.get_ask_price('B')
        amm_ask = (amm.reserve_b / amm.reserve_a) * (1+amm.fee)
        amm_bid = (amm.reserve_b / amm.reserve_a) / (1+amm.fee)
        if amm_ask < market_bid:
            swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid/(1+amm.fee))) / amm.reserve_a
        elif amm_bid > market_ask:
            swap_rate = math.sqrt((amm.reserve_a*amm.reserve_b*market_ask*(1+amm.fee)))/amm.reserve_b - 1
        else:
            swap_rate = 0
        
        info = amm.swap(swap_rate)
        asset_delta = info['asset_delta']
        fee = info['fee']
        asset_in, asset_out = ('A', 'B') if swap_rate < 0 else ('B', 'A')
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
        total_pnl += (market_gain - amm_cost) / market.initial_price if swap_rate != 0 else 0
        total_fee += fee[asset_in] * market.get_ask_price(asset_in) / market.initial_price

        # Record the price discrepancy between AMM and external market
        # market_mid = market.AP / market.BP
        # amm_mid = amm.reserve_b / amm.reserve_a
        # price_distance += abs(market_mid - amm_mid)
        total_volume += 1
        total_transactions += (market_gain + amm_cost) / market.initial_price

        # Update the state of the market and AMM after the trade
        if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
            break

        market.next()
        
    
    # mean_annualized_sigma = np.mean(sigma_values) * np.sqrt(1/config['dt'])

    return total_pnl, total_fee, total_volume, total_transactions


def simulate_with_dynamic_fee_rate(config) -> dict:
    """
    urgent level determines whether the agent places an order or not.
    market competence determines how much percent of arbitrage opportunities will be taken by other traders in the market.
    """
    def find_optimal_fee_rate(min_fee, max_fee, market, amm):
        # Generate 1000 equally spaced fee rates between min_fee and max_fee
        # fee_rates = np.linspace(min_fee, max_fee, 1000)
        fee_rates = [0.008]
        # add 0.003 to the fee rates
        # fee_rates = np.append(fee_rates, 1e-6)
        # Initialize variables to track the best fee rate and the maximum total fee
        best_fee_rate = min_fee
        max_total_fee = -float('inf')  # Start with a very small number
    
        # Loop through each fee rate
        for fee_rate in fee_rates:
            # Set the AMM fee to the current value of fee_rate
            amm.fee = fee_rate
        
            # Calculate market and AMM prices
            market_ask = market.get_ask_price('A') / market.get_bid_price('B')
            market_bid = market.get_bid_price('A') / market.get_ask_price('B')
            amm_ask = (amm.reserve_b / amm.reserve_a) * (1 + amm.fee)
            amm_bid = (amm.reserve_b / amm.reserve_a) / (1 + amm.fee)
        
            # Calculate the swap rate to deplete arbitrage opportunity
            if amm_ask < market_bid:
                swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid / (1 + amm.fee))) / amm.reserve_a
            elif amm_bid > market_ask:
                swap_rate = math.sqrt((amm.reserve_a * amm.reserve_b * market_ask * (1 + amm.fee))) / amm.reserve_b - 1
            else:
                swap_rate = 0
        
            # Perform a simulated swap to calculate the fees
            pre_a = amm.reserve_a
            pre_b = amm.reserve_b
            info = amm.simu_swap(swap_rate)
            post_a = amm.reserve_a
            post_b = amm.reserve_b
            if pre_a != post_a:
                print(f"pre_a: {pre_a} | post_a: {post_a}")
            fee = info['fee']
            
            if amm_ask < market_bid:
                price_distance = market_bid - amm_ask
            elif amm_bid > market_ask:
                price_distance = amm_bid - market_ask
            else:
                price_distance = 0
                   
            # Calculate total fees collected from the swap
            total_fee = (fee['A'] + fee['B'])
  
        
            # If this fee is the highest so far, store the fee rate and the total fee
            if total_fee > max_total_fee:
                max_total_fee = total_fee
                best_fee_rate = fee_rate
            else:
                break
            if max_total_fee > 0:
                amm.fee = best_fee_rate - 0.001
                
                 # Calculate market and AMM prices
                market_ask = market.get_ask_price('A') / market.get_bid_price('B')
                market_bid = market.get_bid_price('A') / market.get_ask_price('B')
                amm_ask = (amm.reserve_b / amm.reserve_a) * (1 + amm.fee)
                amm_bid = (amm.reserve_b / amm.reserve_a) / (1 + amm.fee)
        
                # Calculate the swap rate to deplete arbitrage opportunity
                if amm_ask < market_bid:
                    swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid / (1 + amm.fee))) / amm.reserve_a
                elif amm_bid > market_ask:
                    swap_rate = math.sqrt((amm.reserve_a * amm.reserve_b * market_ask * (1 + amm.fee))) / amm.reserve_b - 1
                else:
                    swap_rate = 0
        
                # Perform a simulated swap to calculate the fees
                pre_a = amm.reserve_a
                pre_b = amm.reserve_b
                info = amm.simu_swap(swap_rate)
                post_a = amm.reserve_a
                post_b = amm.reserve_b
                if pre_a != post_a:
                    print(f"pre_a: {pre_a} | post_a: {post_a}")
                fee = info['fee']
            
                if amm_ask < market_bid:
                    price_distance = market_bid - amm_ask
                elif amm_bid > market_ask:
                    price_distance = amm_bid - market_ask
                else:
                    price_distance = 0
                   
                # Calculate total fees collected from the swap
                new_total_fee = (fee['A'] + fee['B'])
                
                if new_total_fee > max_total_fee:
                    new_fee_rate = amm.fee
                    print(f"old_best_fee_rate: {best_fee_rate}, new_best_fee_rate: {new_fee_rate}")
                    print(f"old_max_total_fee: {max_total_fee}, new_max_total_fee: {new_total_fee}")
                    return new_fee_rate, new_total_fee
  
                
                # fee_rates = np.linspace(best_fee_rate, 0.0005, 100)
                # new_best_fee_rate = best_fee_rate
                # new_max_total_fee = max_total_fee
                # for fee_rate in fee_rates:
                #     # Set the AMM fee to the current value of fee_rate
                #     amm.fee = fee_rate
                
                #     # Calculate market and AMM prices
                #     market_ask = market.get_ask_price('A') / market.get_bid_price('B')
                #     market_bid = market.get_bid_price('A') / market.get_ask_price('B')
                #     amm_ask = (amm.reserve_b / amm.reserve_a) * (1 + amm.fee)
                #     amm_bid = (amm.reserve_b / amm.reserve_a) / (1 + amm.fee)
                
                #     # Calculate the swap rate to deplete arbitrage opportunity
                #     if amm_ask < market_bid:
                #         swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid / (1 + amm.fee))) / amm.reserve_a
                #     elif amm_bid > market_ask:
                #         swap_rate = math.sqrt((amm.reserve_a * amm.reserve_b * market_ask * (1 + amm.fee))) / amm.reserve_b - 1
                #     else:
                #         swap_rate = 0
                
                #     # Perform a simulated swap to calculate the fees
                #     pre_a = amm.reserve_a
                #     pre_b = amm.reserve_b
                #     info = amm.simu_swap(swap_rate)
                #     post_a = amm.reserve_a
                #     post_b = amm.reserve_b
                #     if pre_a != post_a:
                #         print(f"pre_a: {pre_a} | post_a: {post_a}")
                #     fee = info['fee']
                    
                #     if amm_ask < market_bid:
                #         price_distance = market_bid - amm_ask
                #     elif amm_bid > market_ask:
                #         price_distance = amm_bid - market_ask
                #     else:
                #         price_distance = 0
                        
                #     # Calculate total fees collected from the swap
                #     total_fee = (fee['A'] + fee['B'])
                #     if total_fee > new_max_total_fee:
                #         new_max_total_fee = total_fee
                #         new_best_fee_rate = fee_rate
                #     else:
                #         # break
                #         print(f"old_best_fee_rate: {best_fee_rate}, new_best_fee_rate: {new_best_fee_rate}")
                #         return new_best_fee_rate, new_max_total_fee
                    
        # Return the best fee rate
        return best_fee_rate, max_total_fee
    
    initial_fee_rate = 0.003
    
    amm = AMM(fee=initial_fee_rate)
    market = MarketSimulator(sigma=-1,
                             mu=config['mu'],
                             spread=config['spread'],
                             dt=config['dt'],
                             start_price=config['start_price'],
                             steps=config['steps'],
                             seed=123)

    price_distance = 0
    total_pnl = 0
    total_fee = 0
    total_volume = 0
    total_transactions = 0
    mkt_mid = deque(maxlen=1000)
    log_returns = deque(maxlen=999)    
    sum_log_returns = 0
    sum_log_returns_sq = 0
    window_size = 999
    sigma_values = []
    dynamic_fees = []
    vol_based = False

    # Loop over market steps
    for _ in range(int(market.steps)):
        
        mkt_ask = market.get_ask_price('A')
        mkt_bid = market.get_bid_price('B')
        mid_price = (mkt_ask + mkt_bid) / 2
        
        if vol_based:
            if len(mkt_mid) > 0:
                last_price = mkt_mid[-1]
                log_return = np.log(mid_price / last_price)
                sum_log_returns += log_return
                sum_log_returns_sq += log_return ** 2
                log_returns.append(log_return)

                if len(log_returns) == window_size:
                    oldest_return = log_returns[0]
                    sum_log_returns -= oldest_return
                    sum_log_returns_sq -= oldest_return ** 2

            mkt_mid.append(mid_price)

            if len(mkt_mid) == 1000:
                mean_log_return = sum_log_returns / window_size
                variance = (sum_log_returns_sq / window_size) - (mean_log_return ** 2)
                current_sigma = np.sqrt(variance)
                sigma_values.append(current_sigma)
                
                # change AMM fee rate based on the current sigma
                sig_to_sig_coef = 0.7466
                sig_to_sig_const = -0.004
                transition_sigma = 0.1332
                sig_to_fee_rate_const = -0.0006
                sig_to_fee_rate_coef = 0.0308
                raw_sigma = current_sigma
                current_sigma *= np.sqrt(1/config['dt'])
                set_sig = current_sigma * sig_to_sig_coef + sig_to_sig_const
                if set_sig > transition_sigma:
                    amm.fee = 0.0035
                elif set_sig < 0.05:
                    amm.fee = 0.001
                else:
                    amm.fee = set_sig * sig_to_fee_rate_coef + sig_to_fee_rate_const
        else:
            # Set bounds for the fee rate
            bounds = [(0.0005, 0.05)]

            # Use scipy's minimize function to find the optimal fee rate
            # result = opt.minimize(fee_objective, x0=[0.002], args=(market, amm), bounds=bounds, options={'tol': 1e-6})
            optimal_fee_rate, max_total_fee = find_optimal_fee_rate(0.008, 0.0081, market, amm)
            
            if optimal_fee_rate < 0.008:
                print(f"optimal_fee_rate: {optimal_fee_rate}")
            # Optimal fee rate
            # optimal_fee_rate = result.x[0]
            amm.fee = 0.01
        # dynamic_fees.append((amm.fee, current_sigma, set_sig, raw_sigma))
            
        # Get trader observations
        market_ask = market.get_ask_price('A') / market.get_bid_price('B')
        market_bid = market.get_bid_price('A') / market.get_ask_price('B')
        amm_ask = (amm.reserve_b / amm.reserve_a) * (1+amm.fee)
        amm_bid = (amm.reserve_b / amm.reserve_a) / (1+amm.fee)
        if amm_ask < market_bid:
            amm.fee = 0.007
            swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid/(1+amm.fee))) / amm.reserve_a
            # print(f"swap_rate: {swap_rate}")
        elif amm_bid > market_ask:
            amm.fee = 0.007
            swap_rate = math.sqrt((amm.reserve_a*amm.reserve_b*market_ask*(1+amm.fee)))/amm.reserve_b - 1
        else:
            swap_rate = 0
        
        info = amm.swap(swap_rate)
        asset_delta = info['asset_delta']
        fee = info['fee']
        asset_in, asset_out = ('A', 'B') if swap_rate < 0 else ('B', 'A')
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
        total_pnl += (market_gain - amm_cost) / market.initial_price if swap_rate != 0 else 0
        total_fee += (fee['A'] + fee['B']) / market.initial_price
        # if max_total_fee > 0:
        #     print(total_fee)

        # Record the price discrepancy between AMM and external market
        market_mid = market.AP / market.BP
        amm_mid = amm.reserve_b / amm.reserve_a
        price_distance += abs(market_mid - amm_mid)
        total_volume += 1
        total_transactions += (market_gain + amm_cost) / market.initial_price

        # Update the state of the market and AMM after the trade
        if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
            break

        market.next()
        
    
    # mean_annualized_sigma = np.mean(sigma_values) * np.sqrt(1/config['dt'])
    save_folder = config['save_folder']
    os.makedirs(save_folder, exist_ok=True)
    # dynamic_fees = pd.DataFrame(dynamic_fees, columns=['fee', 'current_sigma', 'set_sigma', 'raw_sigma'])
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # dynamic_fees.to_csv(f'{save_folder}/dynamic_fees_{time_stamp}.csv')

    return total_pnl, total_fee, total_volume, price_distance, total_transactions#, mean_annualized_sigma

def simulate_with_rl_dynamic(config) -> dict:
    """
    urgent level determines whether the agent places an order or not.
    market competence determines how much percent of arbitrage opportunities will be taken by other traders in the market.
    """
    
    initial_fee_rate = 0.003
    
    amm = AMM(fee=initial_fee_rate)
    market = MarketSimulator(sigma=-1,
                             mu=config['mu'],
                             spread=config['spread'],
                             dt=config['dt'],
                             start_price=config['start_price'],
                             steps=config['steps'],
                             seed=123)

    price_distance = 0
    total_pnl = 0
    total_fee = 0
    total_volume = 0
    total_transactions = 0
    mkt_mid = deque(maxlen=1000)
    log_returns = deque(maxlen=999)    
    sum_log_returns = 0
    sum_log_returns_sq = 0
    window_size = 999
    sigma_values = []
    dynamic_fees = []
    vol_based = False
    agent = TD3.load('/home/shiftpub/Dynamic_AMM/models/dummy_maker_model/rl_maker_8384000_steps.zip')
    

    # Loop over market steps
    for _ in range(int(market.steps)):
        
        mkt_ask = market.get_ask_price('A')
        mkt_bid = market.get_bid_price('B')
        amm_token_a = amm.reserve_a/amm.initial_a
        amm_token_b = amm.reserve_b/amm.initial_b
        mid_price = (mkt_ask + mkt_bid) / 2
        obs = np.array([mkt_ask, mkt_bid, amm_token_b, amm_token_a])
        action, _states = agent.predict(obs, deterministic=True)     
        amm.fee = action[0]
        dynamic_fees.append((amm.fee))
            
        # Get trader observations
        market_ask = market.get_ask_price('A') / market.get_bid_price('B')
        market_bid = market.get_bid_price('A') / market.get_ask_price('B')
        amm_ask = (amm.reserve_b / amm.reserve_a) * (1+amm.fee)
        amm_bid = (amm.reserve_b / amm.reserve_a) / (1+amm.fee)
        if amm_ask < market_bid:
            swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid/(1+amm.fee))) / amm.reserve_a
        elif amm_bid > market_ask:
            swap_rate = math.sqrt((amm.reserve_a*amm.reserve_b*market_ask*(1+amm.fee)))/amm.reserve_b - 1
        else:
            swap_rate = 0
        
        info = amm.swap(swap_rate)
        asset_delta = info['asset_delta']
        fee = info['fee']
        asset_in, asset_out = ('A', 'B') if swap_rate < 0 else ('B', 'A')
        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
        market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
        total_pnl += (market_gain - amm_cost) / market.initial_price if swap_rate != 0 else 0
        total_fee += (fee['A'] + fee['B']) / market.initial_price
        
        # Record the price discrepancy between AMM and external market
        market_mid = market.AP / market.BP
        amm_mid = amm.reserve_b / amm.reserve_a
        price_distance += abs(market_mid - amm_mid)
        total_volume += 1
        total_transactions += (market_gain + amm_cost) / market.initial_price

        # Update the state of the market and AMM after the trade
        if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
            break

        market.next()
        
    
    # mean_annualized_sigma = np.mean(sigma_values) * np.sqrt(1/config['dt'])
    save_folder = config['save_folder']
    os.makedirs(save_folder, exist_ok=True)
    dynamic_fees = pd.DataFrame(dynamic_fees, columns=['fee_rate'])
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_fees.to_csv(f'{save_folder}/dynamic_fees_{time_stamp}.csv')

    return total_pnl, total_fee, total_volume, price_distance, total_transactions#, mean_annualized_sigma