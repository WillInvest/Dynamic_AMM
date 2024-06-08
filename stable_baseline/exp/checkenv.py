import os
import sys
sys.path.append('../..')

from env.amm_env import ArbitrageEnv
from env.market import GBMPriceSimulator
from env.new_amm import AMM

from stable_baselines3.common.env_checker import check_env


env = ArbitrageEnv(market=GBMPriceSimulator(), amm=AMM())
check_env(env)