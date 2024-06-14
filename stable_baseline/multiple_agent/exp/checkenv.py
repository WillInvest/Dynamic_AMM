import os
import sys
sys.path.append('../../..')

from env.multiAmm import MultiAgentAmm
from env.market import MarketSimulator
from env.new_amm import AMM

from stable_baselines3.common.env_checker import check_env

model_path = '/home/shiftpub/AMM-Python/stable_baseline/single_agent/models/TD3/2024-06-10_10-08-52/agent_seed_0/fee_0.01/sigma_0.2/TD3_910000'

env = MultiAgentAmm(market=MarketSimulator(), amm=AMM(), model_path=model_path)
check_env(env)