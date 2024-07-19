import os
import sys
sys.path.append('../../..')

import time
import numpy as np
import warnings
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
import wandb

from stable_baseline.env.multiAmm import MultiAgentAmm
from stable_baseline.env.market import MarketSimulator
from stable_baseline.env.new_amm import AMM
from stable_baseline.env.callback import WandbCallback

import gymnasium as gym
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

def train(root_path):
    
    TOTAL_STEPS = int(1e6)
    EVALUATE_PER_STEP = int(1e4)
    LEARNING_RATE = 0.0001
    
    for mc in [0.2, 0.4, 0.6, 0.8, 1.0]:
        wandb.init(project="AMM_RL_Trader", entity='willinvest', name=f'mc_{mc:.1f}', config={"market_competence": mc})
        model_dirs = os.path.join(root_path, "models", f"market_competence_{mc:.1f}")
        os.makedirs(model_dirs, exist_ok=True)
                    
        envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(),
                                              amm=AMM(fee=0.0001),
                                              market_competence=mc)) for _ in range(16)]
        env = SubprocVecEnv(envs)
        model = PPO("MlpPolicy", env=env, learning_rate=LEARNING_RATE, gamma=0.95, n_steps=4096, n_epochs=8)
        
        checkpoint_callback = CheckpointCallback(save_freq=EVALUATE_PER_STEP, save_path=model_dirs, name_prefix="rl_model")
        eval_env = Monitor(MultiAgentAmm(market=MarketSimulator(), amm=AMM(), market_competence=mc))
        eval_callback = EvalCallback(eval_env, best_model_save_path=model_dirs, log_path=model_dirs, eval_freq=EVALUATE_PER_STEP)
        wandb_callback = WandbCallback()
        
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, eval_callback, wandb_callback], progress_bar=True)

        wandb.finish()

if __name__ == '__main__':
    ROOT_DIR = '/home/shiftpub/AMM-Python/stable_baseline/multiple_agent'
    train(ROOT_DIR)
