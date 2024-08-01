import os
import sys
# sys.path.append('../../..')

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
    
    TOTAL_STEPS = int(1e6)
    EVALUATE_PER_STEP = int(1e3)
    
    for mc in np.arange(0.02, 0.22, 0.02):
        mc = round(mc, 2)
        wandb.init(project="AMM_RL_Trader_with_random_fee_final",
                   entity='willinvest',
                   name=f'mc_{mc:.2f}',
                   config={"market_competition_level": mc},
                   mode='disabled')
        model_dirs = os.path.join(root_path, "models_trader_final", f"market_competition_level_{mc:.2f}")
        os.makedirs(model_dirs, exist_ok=True)
                    
        envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(seed=seed, steps=500),
                                              amm=AMM(),
                                              market_competition_level=mc)) for seed in range(10)]
        env = SubprocVecEnv(envs)
        model = PPO("MlpPolicy", env=env, n_steps=int(1e3))
        checkpoint_callback = CheckpointCallback(save_freq=EVALUATE_PER_STEP, save_path=model_dirs, name_prefix="rl_model")
        wandb_callback = WandbCallback()
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, wandb_callback], progress_bar=True)
        wandb.finish()

if __name__ == '__main__':
    ROOT_DIR = '/Users/haofu/AMM-Python/models'
    train(ROOT_DIR)
    