from env.market import MarketSimulator
from env.new_amm import AMM
from typing import Dict, List
import numpy as np
from stable_baselines3 import PPO

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

def simulate_with_constant_fee_rate(traders, fee_rate, seed) -> dict:
    """
    urgent level determine whether agent place order or not.
    market competence determine how much percent of arbitrage opportunity will be taken by other traders in the market
    """
    amm = AMM(fee=fee_rate)
    market = MarketSimulator(seed=seed)

    market_steps = 0
    price_distance = 0
    total_pnl = {mc: 0.0 for mc in traders.keys()}
    total_fee = {mc: 0.0 for mc in traders.keys()}
    total_volume = {mc: 0.0 for mc in traders.keys()}
    traders_to_process = list(traders.keys())

    # get the trader observation
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
        trader_actions.sort(reverse=True, key=lambda x: x[0])
            
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
            else:
                # If the highest urgency level is not higher than the fee rate, stop processing
                break
        # record the price discrepancy between AMM and external market
        market_mid = market.AP / market.BP
        amm_mid = amm.reserve_b / amm.reserve_a
        price_distance += abs(market_mid - amm_mid)
                
        # Update the state of the market and AMM after the trade
        market.next()
        market_steps += 1
            
    return total_pnl, total_fee, total_volume, price_distance

def simulate_with_rl_amm(traders, seed, maker_dir) -> Dict[float, List[float]]:
    initial_fee_rate = 0.01
    amm = AMM(fee=initial_fee_rate)
    market = MarketSimulator(seed=seed)
    model_dir = maker_dir
    amm_agent = PPO.load(model_dir)
    
    market_steps = 0
    price_distance = 0
    total_pnl = {mc: 0.0 for mc in traders.keys()}
    total_fee = {mc: 0.0 for mc in traders.keys()}
    total_volume = {mc: 0.0 for mc in traders.keys()}
    dynamic_fees = []

    traders_to_process = list(traders.keys())

    # get the trader observation
    while market_steps < market.steps and min(amm.reserve_a, amm.reserve_b) > amm.initial_shares * 0.2:
        obs = np.array([
            market.get_ask_price('A') / market.get_bid_price('B'),
            market.get_bid_price('A') / market.get_ask_price('B'),
            amm.reserve_b / amm.initial_shares,
            amm.reserve_a / amm.initial_shares
            ], dtype=np.float32)

        action, _ = amm_agent.predict(obs)
        amm.fee = amm.fee_rates[action]
        dynamic_fees.append(amm.fee)
        trader_obs = get_trader_obs(market=market, amm=amm)
        
        trader_obs = get_trader_obs(market=market, amm=amm)
        trader_actions = []
            
        for mc in traders_to_process:
            trader = traders[mc]
            action, _states = trader.predict(trader_obs)
            swap_rate = swap_rates[action[0]] * 0.1
            urgent_level = round(amm.fee_rates[action[1]], 4)
            trader_actions.append((urgent_level, swap_rate, mc))
            
        # Sort by urgent level and get the highest urgency level trader
        trader_actions.sort(reverse=True, key=lambda x: x[0])
        
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
            else:
                # If the highest urgency level is not higher than the fee rate, stop processing
                break
        # record the price discrepancy between AMM and external market
        market_ask, market_bid, amm_ask, amm_bid = trader_obs[:4]
        price_distance += abs((market_ask + market_bid)/2 - (amm_ask + amm_bid)/2)
            
        market.next()
        market_steps += 1

    return total_pnl, total_fee, total_volume, dynamic_fees, price_distance
