import os
import sys
# Get the path to the AMM-Python directory
sys.path.append(f'{os.path.expanduser("~")}/AMM-Python')

import numpy as np
import tensorflow as tf
from collections import deque
import wandb

from env.trader_env import MultiAgentAmm
from env.market import MarketSimulator
from env.new_amm import AMM
from env.callback import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

def train(root_path):
    
    TOTAL_STEPS = int(5e6)
    EVALUATE_PER_STEP = int(1e4)
    
    for mc in np.arange(0.02, 0.22, 0.02):
        mc = round(mc, 2)
        wandb.init(project="AMM_Trader_Train",
                   entity='willinvest',
                   name=f'{os.path.expanduser("~")}_mc_{mc:.2f}',
                   config={"market_competition_level": mc})
        model_dirs = os.path.join(root_path, "trader_model", f"market_competition_level_{mc:.2f}")
        os.makedirs(model_dirs, exist_ok=True)
                    
        envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(seed=seed, steps=500),
                                              amm=AMM(),
                                              market_competition_level=mc)) for seed in range(10)]
        env = SubprocVecEnv(envs)
        model = PPO("MlpPolicy", env=env, n_steps=int(1e3), batch_size=int(1e4))
        checkpoint_callback = CheckpointCallback(save_freq=EVALUATE_PER_STEP, save_path=model_dirs, name_prefix="rl_trader")
        wandb_callback = WandbCallback()
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, wandb_callback], progress_bar=True)
        wandb.finish()

if __name__ == '__main__':
    ROOT_DIR = '/Users/haofu/AMM-Python/models'
    train(ROOT_DIR)
    