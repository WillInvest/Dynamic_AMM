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
from callbacks import EvalCallback, CheckpointCallback, NoiseCallback, TraderWandbCallback

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise


def train(root_path):
    n_envs = 16
    TOTAL_STEPS = n_envs * int(1e7)
    EVALUATE_PER_STEP = TOTAL_STEPS / 10
    CHECKPOINT_PER_STEP = int(1e5)
    
    for mc in np.arange(0.05, 1.05, 0.05):
        mc = round(mc, 2)
        wandb.init(project="Dynamic_AMM_Trader",
                   entity='willinvest',
                   name=f'User-{socket.gethostname()}_mc_{mc:.2f}',
                   config={"market_competition_level": mc})
        model_dirs = os.path.join(root_path, "trader_model", f"market_competition_level_{mc:.2f}")
        log_path = os.path.join(model_dirs, "logs")
        os.makedirs(model_dirs, exist_ok=True)
        envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(seed=seed, steps=5000),
                                              amm=AMM(),
                                              market_competition_level=mc)) for seed in range(n_envs)]
        env = SubprocVecEnv(envs)
        # Define action noise
        n_actions = env.action_space.shape[-1]
        exploration_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))

        model = TD3("MlpPolicy", env, verbose=0, learning_rate=0.0003, train_freq=(n_envs*100, 'step'), gradient_steps=-1, action_noise=exploration_noise)
        checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_PER_STEP, save_path=model_dirs, name_prefix="rl_trader")
        wandb_callback = TraderWandbCallback()
        eval_callback = EvalCallback(env,
                                     best_model_save_path=model_dirs,
                                     log_path=log_path,
                                     eval_freq=EVALUATE_PER_STEP,
                                     n_eval_episodes=n_envs,
                                     deterministic=True)
        noise_schedule_callback = NoiseCallback(start_noise=1.0, end_noise=0.1, total_steps=TOTAL_STEPS)
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, eval_callback, wandb_callback, noise_schedule_callback], progress_bar=True)
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
    