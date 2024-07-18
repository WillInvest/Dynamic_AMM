import os
import sys
sys.path.append('../../..')

from env.multiAmm import MultiAgentAmm
from env.market import MarketSimulator
from env.new_amm import AMM

from stable_baselines3.common.env_checker import check_env


env = MultiAgentAmm(market=MarketSimulator(), amm=AMM())
obs, _ = env.reset()

market_ask, market_bid, amm_ask, amm_bid = obs[0:4]

print(f"market_ask: {market_ask} | market_bid: {market_bid} | amm_ask: {amm_ask} | amm_bid: {amm_bid}")


env.step([0.1, 0.1])

obs, _ = env.reset()
print(f"obs: {obs}")