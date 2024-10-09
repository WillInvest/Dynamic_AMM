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

    # Loop over market steps
    for _ in range(market.steps):
        
        mkt_ask = market.get_ask_price('A')
        mkt_bid = market.get_bid_price('B')
        mid_price = (mkt_ask + mkt_bid) / 2

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
        
    
    mean_annualized_sigma = np.mean(sigma_values) * np.sqrt(1/config['dt'])

    return total_pnl, total_fee, total_volume, price_distance, total_transactions, mean_annualized_sigma


def simulate_with_dynamic_fee_rate(config) -> dict:
    """
    urgent level determines whether the agent places an order or not.
    market competence determines how much percent of arbitrage opportunities will be taken by other traders in the market.
    """
    
    initial_fee_rate = 0.003
    
    amm = AMM(fee=initial_fee_rate)
    market = MarketSimulator(sigma=None,
                             mu=config['mu'],
                             spread=config['spread'],
                             dt=config['dt'],
                             start_price=config['start_price'],
                             steps=config['steps'])

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

    # Loop over market steps
    for _ in range(market.steps):
        
        mkt_ask = market.get_ask_price('A')
        mkt_bid = market.get_bid_price('B')
        mid_price = (mkt_ask + mkt_bid) / 2

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
            dynamic_fees.append((amm.fee, current_sigma, set_sig, raw_sigma))
            
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
        
    
    mean_annualized_sigma = np.mean(sigma_values) * np.sqrt(1/config['dt'])
    save_folder = config['save_folder']
    os.makedirs(save_folder, exist_ok=True)
    dynamic_fees = pd.DataFrame(dynamic_fees, columns=['fee', 'current_sigma', 'set_sigma', 'raw_sigma'])
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_fees.to_csv(f'{save_folder}/dynamic_fees_{time_stamp}.csv')

    return total_pnl, total_fee, total_volume, price_distance, total_transactions, mean_annualized_sigma

# def simulate_with_dynamic_amm(maker_dir, seed, sigma) -> Dict[float, List[float]]:
#     initial_fee_rate = 0.01
#     amm = AMM(fee=initial_fee_rate)
#     market = MarketSimulator(seed=seed, sigma=sigma)
#     amm_agent = TD3.load(maker_dir)
    
#     price_distance = 0
#     total_pnl = 0
#     total_fee = 0
#     total_volume = 0
#     total_transactions = 0
#     dynamic_fees = []

#     # Run the simulation for the given market steps
#     for _ in range(market.steps):
#         # Get AMM agent's observation and predict new fee rate
#         obs = np.array([
#             market.get_ask_price('A') / market.get_bid_price('B'),
#             market.get_bid_price('A') / market.get_ask_price('B'),
#             amm.reserve_b / amm.initial_shares,
#             amm.reserve_a / amm.initial_shares
#         ], dtype=np.float32)

#         action, _ = amm_agent.predict(obs)
#         amm.fee = np.round(action[0], 4)
#         dynamic_fees.append(amm.fee)
        
#         # Get trader observations
#         market_ask = market.get_ask_price('A') / market.get_bid_price('B')
#         market_bid = market.get_bid_price('A') / market.get_ask_price('B')
#         amm_ask = (amm.reserve_b / amm.reserve_a) * (1+amm.fee)
#         amm_bid = (amm.reserve_b / amm.reserve_a) / (1+amm.fee)
#         if amm_ask < market_bid:
#             swap_rate = 1 - math.sqrt(amm.reserve_a * amm.reserve_b / (market_bid/(1+amm.fee))) / amm.reserve_a
#         elif amm_bid > market_ask:
#             swap_rate = math.sqrt((amm.reserve_a*amm.reserve_b*market_ask*(1+amm.fee)))/amm.reserve_b - 1
#         else:
#             swap_rate = 0
        
#         info = amm.swap(swap_rate)
#         asset_delta = info['asset_delta']
#         fee = info['fee']
#         asset_in, asset_out = ('A', 'B') if swap_rate < 0 else ('B', 'A')
#         amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
#         market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
#         total_pnl += (market_gain - amm_cost) / market.initial_price if swap_rate != 0 else 0
#         total_fee += (fee['A'] + fee['B']) / market.initial_price

#         # Record the price discrepancy between AMM and external market
#         market_mid = market.AP / market.BP
#         amm_mid = amm.reserve_b / amm.reserve_a
#         price_distance += abs(market_mid - amm_mid)

#         # Update market state
#         market.next()

#         # Check if AMM reserves are too low to continue
#         if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
#             break

#     return total_pnl, total_fee, total_volume, dynamic_fees, price_distance, total_transactions