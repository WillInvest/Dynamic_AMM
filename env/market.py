import os
import sys
import socket

# Get the path to the AMM-Python directory
sys.path.append(f'{os.path.expanduser("~")}/AMM-Python')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from env.new_amm import AMM
import math


import numpy as np

class MarketSimulator:
    def __init__(self,
                 start_price=50000,
                 mu=0.0001,
                 sigma=None,
                 dt=1,
                 steps=5000,
                 spread=0.01,
                 seed=None):
        
        self.initial_price = start_price
        self.sigma = sigma
        self.AP = start_price
        self.BP = start_price
        self.current_price = start_price
        self.mu = mu
        self.random_sigma = True if sigma is None else False
        self.spread = spread
        self.dt = dt
        self.index = 0
        self.steps = steps

        # Initialize random number generator, only if seed is provided
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()  # Random behavior without a seed

    def get_random_sigma(self):
        return self.rng.choice([0.005, 0.006, 0.007, 0.008, 0.009, 0.01])

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
        shock1 = self.rng.normal()
        shock2 = self.rng.normal()

        # Update the current price using the GBM formula
        self.AP *= np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * shock1)
        self.BP *= np.exp(
            (self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * shock2)
    
    def reset(self):
        self.AP = self.initial_price
        self.BP = self.initial_price
        self.index = 0

        
        
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
    

    
if __name__ == '__main__':
  
    # from tqdm import tqdm
    
    # old_50 = [9, 13, 8, 17, 18, 16, 10, 18, 12, 18, 8, 17, 11, 13, 9, 21, 11, 11, 14, 18, 15, 8, 18, 15, 11, 12, 10, 12, 10, 13, 14, 9, 3, 10, 6, 10, 10, 15, 13, 15, 10, 15, 6, 9, 10, 10, 10, 7, 15, 3, 11, 14, 22, 21, 9, 22, 14, 11, 13, 14, 16, 13, 13, 19, 15, 10, 7, 8, 14, 14, 13, 14, 7, 13, 12, 10, 14, 22, 12, 15, 17, 4, 14, 18, 15, 13, 7, 13, 20, 13, 17, 8, 8, 15, 11, 13, 15, 8, 15, 15]
    # old_500 = [149, 157, 128, 149, 155, 130, 129, 117, 167, 140, 115, 187, 138, 131, 128, 155, 133, 144, 126, 122, 144, 120, 153, 135, 139, 127, 132, 146, 148, 166, 141, 126, 137, 128, 136, 142, 141, 172, 151, 153, 158, 154, 138, 148, 144, 145, 130, 161, 152, 126, 126, 139, 151, 159, 142, 154, 137, 130, 150, 145, 123, 134, 144, 146, 116, 168, 124, 136, 131, 144, 164, 158, 129, 142, 128, 155, 152, 164, 138, 144, 138, 140, 136, 122, 141, 161, 113, 131, 135, 146, 157, 139, 115, 135, 165, 123, 130, 141, 131, 135]
    # old_5000 = [1417, 1440, 1461, 1440, 1430, 1526, 1423, 1428, 1449, 1418, 1348, 1511, 1445, 1407, 1422, 1439, 1457, 1386, 1503, 1348, 1385, 1414, 1505, 1407, 1560, 1356, 1490, 1458, 1578, 1458, 1386, 1384, 1449, 1377, 1398, 1460, 1417, 1476, 1517, 1473, 1484, 1440, 1379, 1497, 1450, 1388, 1483, 1521, 1545, 1453, 1388, 1439, 1426, 1411, 1453, 1467, 1424, 1409, 1489, 1510, 1468, 1413, 1448, 1416, 1355, 1461, 1358, 1398, 1417, 1360, 1398, 1433, 1374, 1540, 1467, 1390, 1384, 1498, 1415, 1452, 1356, 1376, 1400, 1378, 1438, 1440, 1404, 1402, 1427, 1400, 1500, 1441, 1408, 1505, 1503, 1402, 1407, 1411, 1443, 1475]
    # trade_counts = []

    # for dt in [1, 1/60, 1/360]:
    #     for steps in [50, 500, 5000]:
    #         for seed in tqdm(range(100), desc=f'dt: {dt}, steps: {steps}'):    
    #             sigma = None
    #             market = MarketSimulator(start_price=50000, deterministic=False, steps=steps, seed=seed, sigma=sigma, dt=dt)
    #             amm = AMM(fee=0.0005)
    #             old_amm_asks = []
    #             old_amm_bids = []
    #             new_amm_asks = []
    #             new_amm_bids = []
    #             mkt_asks = []
    #             mkt_bids = []
    #             APs = []
    #             BPs = []
    #             trade_count = 0

    #             for _ in range(steps):
    #                 mkt_asks.append(market.get_ask_price('A') / market.get_bid_price('B'))
    #                 mkt_bids.append(market.get_bid_price('A') / market.get_ask_price('B'))
    #                 old_amm_ask, old_amm_bid, new_amm_ask, new_amm_bid, if_trade = simulate_amm_with_market(amm, market)
    #                 old_amm_asks.append(old_amm_ask)
    #                 old_amm_bids.append(old_amm_bid)
    #                 new_amm_asks.append(new_amm_ask)
    #                 new_amm_bids.append(new_amm_bid)
    #                 APs.append(market.AP)
    #                 BPs.append(market.BP)
    #                 trade_count += if_trade
    #                 market.next()    
            
    #             trade_counts.append({
    #                 'dt': dt,
    #                 'total_steps': steps,
    #                 'seed': seed,
    #                 'trade_count': trade_count
    #             })
                
    # trade_counts.extend([{'dt': 'old', 'total_steps': 50, 'seed': seed, 'trade_count': trade_count} for seed, trade_count in enumerate(old_50)])
    # trade_counts.extend([{'dt': 'old', 'total_steps': 500, 'seed': seed, 'trade_count': trade_count} for seed, trade_count in enumerate(old_500)])
    # trade_counts.extend([{'dt': 'old', 'total_steps': 5000, 'seed': seed, 'trade_count': trade_count} for seed, trade_count in enumerate(old_5000)])
    
    # pd.DataFrame(trade_counts).to_csv('trade_counts.csv')
    
    trade_counts = []
    
    for seed in range(100):
    
        steps = 50
        sigma = None
        market = MarketSimulator(start_price=50000, deterministic=False, steps=steps, seed=seed, sigma=sigma)
        amm = AMM(fee=0.0005)
        old_amm_asks = []
        old_amm_bids = []
        new_amm_asks = []
        new_amm_bids = []
        mkt_asks = []
        mkt_bids = []
        APs = []
        BPs = []
        trade_count = 0

        for _ in range(steps):
            mkt_asks.append(market.get_ask_price('A') / market.get_bid_price('B'))
            mkt_bids.append(market.get_bid_price('A') / market.get_ask_price('B'))
            old_amm_ask, old_amm_bid, new_amm_ask, new_amm_bid, if_trade = simulate_amm_with_market(amm, market)
            old_amm_asks.append(old_amm_ask)
            old_amm_bids.append(old_amm_bid)
            new_amm_asks.append(new_amm_ask)
            new_amm_bids.append(new_amm_bid)
            APs.append(market.AP)
            BPs.append(market.BP)
            trade_count += if_trade
            market.next()    
            
        trade_counts.append(trade_count)
    
    print(trade_counts)
    
    # Plotting the stair-step graph for the AMM old and new prices
    plt.figure(figsize=(15, 10))
    # plt.plot(np.arange(steps), APs, label='Asset A Price', linestyle='-', color='blue')
    # plt.plot(np.arange(steps), BPs, label='Asset B Price', linestyle='-', color='green')
    plt.plot(np.arange(steps), mkt_asks, label='Market Ask Price', linestyle='-', color='red')
    plt.plot(np.arange(steps), mkt_bids, label='Market Bid Price', linestyle='-', color='purple')
    # Plot old ask prices
    # plt.step(np.arange(steps), old_amm_asks, where='post', label='Old AMM Ask Price', linestyle='--', color='blue')
    # Plot new ask prices
    plt.step(np.arange(steps), new_amm_asks, where='post', label='New AMM Ask Price', linestyle='-', color='blue')
    # Plot old bid prices
    # plt.step(np.arange(steps), old_amm_bids, where='post', label='Old AMM Bid Price', linestyle='--', color='green')
    # Plot new bid prices
    plt.step(np.arange(steps), new_amm_bids, where='post', label='New AMM Bid Price', linestyle='-', color='green')
    # Adding labels and title
    plt.title(f"AMM Old and New Prices (Stair-Step Plot) : {steps} steps, total arbitrage count: {trade_count}", fontsize=16)
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    
    # Adding grid and legend
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig("stair-step-plot.png")
    
    # Show plot
    plt.show()