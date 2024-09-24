from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO, TD3
from tqdm import tqdm

swap_rates = [np.round(rate, 2) for rate in np.arange(-1.0, 1.1, 0.1) if np.round(rate, 2) != 0]

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
    
def calculate_pnl(info, swap_rate, market:MarketSimulator) -> tuple:
    if swap_rate < 0:
        asset_in, asset_out = 'A', 'B'
    else:
        asset_in, asset_out = 'B', 'A'
    asset_delta = info['asset_delta']
    fee = info['fee']
    amm_cost = (asset_delta[asset_in] + fee[asset_in]) * market.get_ask_price(asset_in)
    market_gain = (abs(asset_delta[asset_out])) * market.get_bid_price(asset_out)
    pnl = (market_gain - amm_cost) / market.initial_price if swap_rate != 0 else 0  
    fees = (fee['A'] + fee['B']) / market.initial_price
    return pnl, fees, amm_cost, market_gain

def simulate_with_constant_fee_rate(traders, fee_rate, seed, sigma) -> dict:
    """
    urgent level determines whether the agent places an order or not.
    market competence determines how much percent of arbitrage opportunities will be taken by other traders in the market.
    """
    amm = AMM(fee=fee_rate)
    market = MarketSimulator(seed=seed, sigma=sigma)

    price_distance = 0
    total_pnl = {mc: 0.0 for mc in traders.keys()}
    total_fee = {mc: 0.0 for mc in traders.keys()}
    total_volume = {mc: 0.0 for mc in traders.keys()}
    total_transactions = {mc: 0.0 for mc in traders.keys()}
    traders_to_process = list(traders.keys())

    # Store trader transactions
    trader_transaction = {
        'amm_ask': [],
        'amm_bid': [],
        'market_ask': [],
        'market_bid': [],
        'amm_fee_rate': [],
        'trader_swap_rate': [],
        'trader_urgent_level': [],
        'trader_mc': []
    }

    # Loop over market steps
    for _ in tqdm(range(market.steps), desc=f'fee_rate:{fee_rate}, sigma_{sigma}'):
        # Get the trader observation
        trader_obs = get_trader_obs(market=market, amm=amm)
        market_ask, market_bid, amm_ask, amm_bid, amm_fee_rate = trader_obs

        # Collect trader actions and store data
        trader_actions = []
        for mc in traders_to_process:
            trader = traders[mc]
            swap_rate, urgent_level = trader.predict(trader_obs)[0]  # Get action (ignoring _states)
            trader_actions.append((urgent_level, swap_rate, mc))

        # Collect common data for all traders only once
        num_traders = len(traders_to_process)
        trader_transaction['amm_ask'].extend([amm_ask] * num_traders)
        trader_transaction['amm_bid'].extend([amm_bid] * num_traders)
        trader_transaction['market_ask'].extend([market_ask] * num_traders)
        trader_transaction['market_bid'].extend([market_bid] * num_traders)
        trader_transaction['amm_fee_rate'].extend([amm_fee_rate] * num_traders)
        trader_transaction['trader_swap_rate'].extend([action[1] for action in trader_actions])
        trader_transaction['trader_urgent_level'].extend([action[0] for action in trader_actions])
        trader_transaction['trader_mc'].extend(traders_to_process)

        # Sort by urgent level (descending) to get the highest urgency level trader first
        trader_actions.sort(reverse=True, key=lambda x: x[0])

        # Execute trades
        for urgent_level, swap_rate, mc in trader_actions:
            if urgent_level < amm.fee:
                break  # Stop processing if the highest urgency level is lower than the fee rate

            # Check profit availability by simulating the swap
            simu_info = amm.simu_swap(swap_rate)
            simu_pnl, _, _, _ = calculate_pnl(simu_info, swap_rate, market)

            if simu_pnl > 0:
                info = amm.swap(swap_rate)
                pnl, fees, amm_cost, market_gain = calculate_pnl(info, swap_rate, market)
                total_pnl[mc] += pnl
                total_fee[mc] += fees
                total_volume[mc] += 1
                total_transactions[mc] += (amm_cost + market_gain)

        # Record the price discrepancy between AMM and external market
        market_mid = market.AP / market.BP
        amm_mid = amm.reserve_b / amm.reserve_a
        price_distance += abs(market_mid - amm_mid)

        # Update the state of the market and AMM after the trade
        if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
            break

        market.next()

    return total_pnl, total_fee, total_volume, price_distance, total_transactions, trader_transaction

def simulate_with_rl_amm(traders, seed, maker_dir, sigma) -> Dict[float, List[float]]:
    initial_fee_rate = 0.01
    amm = AMM(fee=initial_fee_rate)
    market = MarketSimulator(seed=seed, sigma=sigma)
    amm_agent = TD3.load(maker_dir)
    
    price_distance = 0
    total_pnl = {mc: 0.0 for mc in traders.keys()}
    total_fee = {mc: 0.0 for mc in traders.keys()}
    total_volume = {mc: 0.0 for mc in traders.keys()}
    total_transactions = {mc: 0.0 for mc in traders.keys()}
    dynamic_fees = []

    traders_to_process = list(traders.keys())

    # Initialize trader transaction dictionary
    trader_transaction = {
        'amm_ask': [],
        'amm_bid': [],
        'market_ask': [],
        'market_bid': [],
        'amm_fee_rate': [],
        'trader_swap_rate': [],
        'trader_urgent_level': [],
        'trader_mc': []
    }

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
        trader_obs = get_trader_obs(market=market, amm=amm)
        market_ask, market_bid, amm_ask, amm_bid, amm_fee_rate = trader_obs
        trader_actions = []

        # Collect trader actions and store data
        for mc in traders_to_process:
            trader = traders[mc]
            swap_rate, urgent_level = trader.predict(trader_obs)[0]  # Get action (ignoring _states)
            trader_actions.append((urgent_level, swap_rate, mc))

        # Collect common data for all traders only once
        num_traders = len(traders_to_process)
        trader_transaction['amm_ask'].extend([amm_ask] * num_traders)
        trader_transaction['amm_bid'].extend([amm_bid] * num_traders)
        trader_transaction['market_ask'].extend([market_ask] * num_traders)
        trader_transaction['market_bid'].extend([market_bid] * num_traders)
        trader_transaction['amm_fee_rate'].extend([amm_fee_rate] * num_traders)
        trader_transaction['trader_swap_rate'].extend([action[1] for action in trader_actions])
        trader_transaction['trader_urgent_level'].extend([action[0] for action in trader_actions])
        trader_transaction['trader_mc'].extend(traders_to_process)

        # Sort by urgent level (descending) to get the highest urgency level trader first
        trader_actions.sort(reverse=True, key=lambda x: x[0])

        # Execute trades
        for urgent_level, swap_rate, mc in trader_actions:
            if urgent_level < amm.fee:
                break  # Stop processing if the highest urgency level is lower than the fee rate

            # Check profit availability by simulating the swap
            simu_info = amm.simu_swap(swap_rate)
            simu_pnl, _, _, _ = calculate_pnl(simu_info, swap_rate, market)

            if simu_pnl > 0:
                info = amm.swap(swap_rate)
                pnl, fees, amm_cost, market_gain = calculate_pnl(info, swap_rate, market)
                total_pnl[mc] += pnl
                total_fee[mc] += fees
                total_volume[mc] += 1
                total_transactions[mc] += (amm_cost + market_gain)

        # Record the price discrepancy between AMM and external market
        price_distance += abs((market_ask + market_bid) / 2 - (amm_ask + amm_bid) / 2)

        # Update market state
        market.next()

        # Check if AMM reserves are too low to continue
        if min(amm.reserve_a, amm.reserve_b) < amm.initial_shares * 0.2:
            break

    return total_pnl, total_fee, total_volume, dynamic_fees, price_distance, total_transactions, trader_transaction