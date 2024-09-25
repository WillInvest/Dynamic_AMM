from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO, TD3
from tqdm import tqdm
import math

def simulate_with_constant_fee_rate(fee_rate, seed, sigma) -> dict:
    """
    urgent level determines whether the agent places an order or not.
    market competence determines how much percent of arbitrage opportunities will be taken by other traders in the market.
    """
    amm = AMM(fee=fee_rate)
    market = MarketSimulator(seed=seed, sigma=sigma)

    price_distance = 0
    total_pnl = 0
    total_fee = 0
    total_volume = 0
    total_transactions = 0

    # Loop over market steps
    for _ in tqdm(range(market.steps), desc=f'fee_rate:{fee_rate}, sigma_{sigma}'):
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

    return total_pnl, total_fee, total_volume, price_distance, total_transactions

def simulate_with_rl_amm(maker_dir, seed, sigma) -> Dict[float, List[float]]:
    initial_fee_rate = 0.01
    amm = AMM(fee=initial_fee_rate)
    market = MarketSimulator(seed=seed, sigma=sigma)
    amm_agent = TD3.load(maker_dir)
    
    price_distance = 0
    total_pnl = 0
    total_fee = 0
    total_volume = 0
    total_transactions = 0
    dynamic_fees = []

    # Run the simulation for the given market steps
    for _ in tqdm(range(market.steps), desc=f'RL Simulation, sigma={sigma}'):
        # Get AMM agent's observation and predict new fee rate
        obs = np.array([
            market.get_ask_price('A') / market.get_bid_price('B'),
            market.get_bid_price('A') / market.get_ask_price('B'),
            amm.reserve_b / amm.initial_shares,
            amm.reserve_a / amm.initial_shares
        ], dtype=np.float32)

        action, _ = amm_agent.predict(obs)
        amm.fee = np.round(action[0], 4)
        dynamic_fees.append(amm.fee)
        
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

        # Update market state
        market.next()

        # Check if AMM reserves are too low to continue
        if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
            break

    return total_pnl, total_fee, total_volume, dynamic_fees, price_distance, total_transactions