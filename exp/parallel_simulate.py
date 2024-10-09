import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO
import multiprocessing
import time
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
swap_rates = [np.round(rate, 2) for rate in np.arange(-1.0, 1.1, 0.1) if np.round(rate, 2) != 0]

# for sort and avoid lambda to use multiple process
def get_urgent_level(trader_action):
    return trader_action[0]

def get_trader_obs(market, amm) -> np.array:
    """
    return 5 states
    1) market ask price
    2) market bid price
    3) amm ask price
    4) amm bid price
    5) amm fee rate
    """
    return np.array([
        market.get_ask_price('A') / market.get_bid_price('B'),
        market.get_bid_price('A') / market.get_ask_price('B'),
        (amm.reserve_b / amm.reserve_a) * (1+amm.fee),
        (amm.reserve_b / amm.reserve_a) / (1+amm.fee),
        amm.fee
        ], dtype=np.float32)

def simulate_with_constant_fee_rate(trader_paths, fee_rates, sigmas, seed):
    """
    Simulate across multiple fee rates and sigmas for a single seed.
    
    Parameters:
    trader_paths: dict
        Dictionary of trader model paths.
    fee_rates: list
        List of fee rates to be tested.
    sigmas: list
        List of sigma values (volatility) to be tested.
    seed: int
        Seed value for random number generation.
    
    Returns:
    dict
        Dictionary containing results for each combination of fee rate and sigma.
    """
    # Load all traders once per seed
    traders = {}
    for mc, path in trader_paths.items():
        traders[mc] = PPO.load(path)
    print(f"seed:{seed}")
    seed = seed + int(time.time()) + os.getpid()
    
    results = {}
    
    for sigma in sigmas:
        logging.info(f"Starting simulations for sigma: {sigma}, seed: {seed}")
        
        for fee_rate in fee_rates:
            logging.info(f"Starting simulation for fee rate: {fee_rate:.4f}, sigma: {sigma}, seed: {seed}")
            
            amm = AMM(fee=fee_rate)
            market = MarketSimulator(seed=seed, sigma=sigma)

            market_steps = 0
            price_distance = 0
            total_pnl = {mc: 0.0 for mc in traders.keys()}
            total_fee = {mc: 0.0 for mc in traders.keys()}
            total_volume = {mc: 0.0 for mc in traders.keys()}
            total_transactions = {mc: 0.0 for mc in traders.keys()}
            traders_to_process = list(traders.keys())

            # Run simulation for each fee rate
            while market_steps < market.steps and min(amm.reserve_a, amm.reserve_b) > amm.initial_shares * 0.2:

                trader_obs = get_trader_obs(market=market, amm=amm)
                trader_actions = []
                    
                for mc in traders_to_process:
                    trader = traders[mc]
                    action, _states = trader.predict(trader_obs)
                    swap_rate = swap_rates[action[0]] * 0.1
                    urgent_level = round(amm.fee_rates[action[1]], 4)
                    trader_actions.append((urgent_level, swap_rate, mc))

                # Sort by urgent level and get the highest urgency level trader
                trader_actions.sort(reverse=True, key=get_urgent_level)
                    
                # Execute the trade if the urgency level is higher than the fee rate
                for action in trader_actions:
                    urgent_level, swap_rate, mc = action
                    if urgent_level >= amm.fee:
                        info = amm.swap(swap_rate)
                        if swap_rate < 0:
                            asset_in, asset_out = 'A', 'B'
                        else:
                            asset_in, asset_out = 'B', 'A'
                        asset_delta = info['asset_delta']
                        fee = info['fee']
                        amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
                        market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
                        pnl = (market_gain - amm_cost)/market.initial_price if swap_rate != 0 else 0    
                        if pnl > 0:
                            total_pnl[mc] += pnl
                            total_fee[mc] += (fee['A'] + fee['B'])
                            total_volume[mc] += 1
                            total_transactions[mc] += (amm_cost + market_gain)
                    else:
                        # If the highest urgency level is not higher than the fee rate, stop processing
                        break
                # Record the price discrepancy between AMM and external market
                market_mid = market.AP / market.BP
                amm_mid = amm.reserve_b / amm.reserve_a
                price_distance += abs(market_mid - amm_mid)
                        
                # Update the state of the market and AMM after the trade
                market.next()
                market_steps += 1
            
            # Store the results for this fee rate and sigma
            results[(fee_rate, sigma)] = {
                'total_pnl': total_pnl,
                'total_fee': total_fee,
                'total_volume': total_volume,
                'price_distance': price_distance,
                'total_transactions': total_transactions
            }

            logging.info(f"Finished simulation for fee rate: {fee_rate:.4f}, sigma: {sigma}, seed: {seed}")
    
    return results

def simulate_with_rl_amm(trader_paths, maker_dir, sigmas, seed):
    """
    Simulate across multiple fee rates and sigmas for a single seed using a reinforcement learning AMM.
    
    Parameters:
    trader_paths: dict
        Dictionary of trader model paths.
    fee_rates: list
        List of fee rates to be tested.
    sigmas: list
        List of sigma values (volatility) to be tested.
    seed: int
        Seed value for random number generation.
    maker_dir: str
        Directory of the trained RL model for the AMM.
    
    Returns:
    dict
        Dictionary containing results for each combination of fee rate and sigma.
    """
    # Load all traders once per seed
    traders = {}
    for mc, path in trader_paths.items():
        traders[mc] = PPO.load(path)
    
    # Load the AMM agent once per seed
    amm_agent = PPO.load(maker_dir)
    
    seed = seed + int(time.time()) + os.getpid()
    
    results = {}
    
    for sigma in sigmas:
        logging.info(f"Starting simulations for sigma: {sigma}, seed: {seed}")
        initial_fee_rate = 0.01  # Start with a base fee rate
        amm = AMM(fee=initial_fee_rate)
        market = MarketSimulator(seed=seed, sigma=sigma)

        market_steps = 0
        price_distance = 0
        total_pnl = {mc: 0.0 for mc in traders.keys()}
        total_fee = {mc: 0.0 for mc in traders.keys()}
        total_volume = {mc: 0.0 for mc in traders.keys()}
        total_transactions = {mc: 0.0 for mc in traders.keys()}
        dynamic_fees = []

        traders_to_process = list(traders.keys())

        # Run simulation for each fee rate
        while market_steps < market.steps and min(amm.reserve_a, amm.reserve_b) > amm.initial_shares * 0.2:
            obs = np.array([
                market.get_ask_price('A') / market.get_bid_price('B'),
                market.get_bid_price('A') / market.get_ask_price('B'),
                amm.reserve_b / amm.initial_shares,
                amm.reserve_a / amm.initial_shares
            ], dtype=np.float32)

            # Predict the action (fee rate) using the RL AMM agent
            action, _ = amm_agent.predict(obs)
            amm.fee = amm.fee_rates[action]
            dynamic_fees.append(amm.fee)

            trader_obs = get_trader_obs(market=market, amm=amm)
            trader_actions = []
            
            for mc in traders_to_process:
                trader = traders[mc]
                action, _states = trader.predict(trader_obs)
                swap_rate = swap_rates[action[0]] * 0.1
                urgent_level = round(amm.fee_rates[action[1]], 4)
                trader_actions.append((urgent_level, swap_rate, mc))

            # Sort by urgent level and get the highest urgency level trader
            trader_actions.sort(reverse=True, key=get_urgent_level)
            
            # Execute the trade if the urgency level is higher than the fee rate
            for action in trader_actions:
                urgent_level, swap_rate, mc = action
                if urgent_level >= amm.fee:
                    info = amm.swap(swap_rate)
                    if swap_rate < 0:
                        asset_in, asset_out = 'A', 'B'
                    else:
                        asset_in, asset_out = 'B', 'A'
                    asset_delta = info['asset_delta']
                    fee = info['fee']
                    amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
                    market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
                    pnl = (market_gain - amm_cost)/market.initial_price if swap_rate != 0 else 0   
                    if pnl > 0: 
                        total_pnl[mc] += pnl
                        total_fee[mc] += (fee['A'] + fee['B'])
                        total_volume[mc] += 1
                        total_transactions[mc] += (amm_cost + market_gain)
                else:
                    # If the highest urgency level is not higher than the fee rate, stop processing
                    break
            
            # Record the price discrepancy between AMM and external market
            market_ask, market_bid, amm_ask, amm_bid = trader_obs[:4]
            price_distance += abs((market_ask + market_bid)/2 - (amm_ask + amm_bid)/2)
            
            market.next()
            market_steps += 1

        # Store the results for this fee rate and sigma
        results[sigma] = {
            'total_pnl': total_pnl,
            'total_fee': total_fee,
            'total_volume': total_volume,
            'dynamic_fees': dynamic_fees,
            'price_distance': price_distance,
            'total_transactions': total_transactions
        }

        logging.info(f"Finished simulation, sigma: {sigma}, seed: {seed}")
    
    return results


def parallel_simulate_with_constant_fee_rate(trader_paths, fee_rates,  sigmas, iterations):
    tasks = [(trader_paths, fee_rates, sigmas, seed) for seed in range(iterations)]
    
    with multiprocessing.Pool(processes=30) as pool:  # Adjust the number of processes as needed
        results = list(tqdm(pool.starmap(simulate_with_constant_fee_rate, tasks), total=len(tasks), desc="Simulating with constant fee rates"))
    
    return results

def parallel_simulate_with_rl_amm(trader_paths, maker_dir, sigmas, iterations):
    tasks = [(trader_paths, maker_dir, sigmas, seed) for seed in range(iterations)]
    
    with multiprocessing.Pool(processes=30) as pool:  # Adjust the number of processes as needed
        results = list(tqdm(pool.starmap(simulate_with_rl_amm, tasks), total=len(tasks), desc="Simulating with RL AMM"))
    
    return results

if __name__ == "__main__":
    import dill as pickle
    trader_dir = f'{os.path.expanduser("~")}/Dynamic_AMM/models/trader_model'
    amm = AMM(fee=0.05)
    market = MarketSimulator(seed=1, sigma=0.2)
    traders = {}
    for mc in np.arange(0.02, 0.22, 0.02):
        model_path = os.path.join(trader_dir, f'market_competition_level_{mc:.2f}', 'best_model.zip')
        if os.path.exists(model_path):
            traders[mc] = PPO.load(model_path)
            print(f"Loaded model for market competition level {mc:.2f}")

    try:
        pickle.dumps(traders)
        print("traders object is dillable (pickleable with dill)")
    except Exception as e:
        print("Dill Error in traders:", e)