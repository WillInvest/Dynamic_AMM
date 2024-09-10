import os
import sys
import socket
import gc

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

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def train(root_path):
    
    TOTAL_STEPS = int(1e6)
    EVALUATE_PER_STEP = int(1e2)
    CHECKPOINT_PER_STEP = int(1e3)
    
    for mc in np.arange(0.35, 1.05, 0.05):
        mc = round(mc, 2)
        wandb.init(project="Dynamic_AMM_Trader",
                   entity='willinvest',
                   name=f'User-{socket.gethostname()}_mc_{mc:.2f}',
                   config={"market_competition_level": mc})
        model_dirs = os.path.join(root_path, "trader_model", f"market_competition_level_{mc:.2f}")
        log_path = os.path.join(model_dirs, "logs")
        os.makedirs(model_dirs, exist_ok=True)
        n_envs = 32
        envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(seed=seed, steps=5000),
                                              amm=AMM(),
                                              market_competition_level=mc)) for seed in range(n_envs)]
        env = SubprocVecEnv(envs)
        model = TD3("MlpPolicy", env, verbose=1, learning_rate=0.0003, train_freq=(1, 'step'), gradient_steps=-1)
        checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_PER_STEP, save_path=model_dirs, name_prefix="rl_trader")
        wandb_callback = WandbCallback()
        eval_callback = EvalCallback(env,
                                     best_model_save_path=model_dirs,
                                     log_path=log_path,
                                     eval_freq=EVALUATE_PER_STEP,
                                     n_eval_episodes=n_envs,
                                     deterministic=True,
                                     render=False)
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, eval_callback, wandb_callback], progress_bar=True)
        wandb.finish()
        gc.collect()

if __name__ == '__main__':
    ROOT_DIR = f'{os.path.expanduser("~")}/AMM-Python/models'
    train(ROOT_DIR)
    
    
    
    """
    1. calculate the mean and variance of BitCoin and replace it with the GBM parameters
    2. use rule-based agent with different parameters (amount of arbitrage to consume)
    3. amm_env TODO: create a fake AMM to test whether the swap will generate positive PnL
    4. check how much arbitrage left after traders place orders
    
    """
    