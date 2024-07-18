import os
import sys
import time
import numpy as np
import warnings
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm
sys.path.append('../../..')
from stable_baseline.env.multiAmm import MultiAgentAmm
from stable_baseline.env.market import MarketSimulator
from stable_baseline.env.new_amm import AMM

import gymnasium as gym
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

import wandb

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        # Log the desired metrics to wandb
        wandb.log({
            "reward_1": self.locals['infos'][0]['rew1'],
            "reward_2": self.locals['infos'][0]['rew2'],
            "fee_1": self.locals['infos'][0]['fee1'],
            "fee_2": self.locals['infos'][0]['fee2'],
            "gas_fee_1": self.locals['infos'][0]['gas_fee1'],
            "gas_fee_2": self.locals['infos'][0]['gas_fee2'],
            "swap_rate_1": self.locals['infos'][0]['swap_rate1'],
            "swap_rate_2": self.locals['infos'][0]['swap_rate2'],
            "total_rewards": self.locals['infos'][0]['total_rewards'],
            "total_gas": self.locals['infos'][0]['total_gas'],
            "amm_fee": self.training_env.get_attr('amm')[0].fee,
            "market_sigma": self.training_env.get_attr('market')[0].sigma
        })
        return True

def train(root_path):
    
    TOTAL_STEPS = int(1e8)
    EVALUATE_PER_STEP = int(1e4)
    LEARNING_RATE = 0.0001
    
    for mc in [0.2, 0.4, 0.6, 0.8, 1.0]:
        wandb.init(project="AMM_RL_Trader", entity='willinvest', name=f'mc_{mc:.1f}', config={"market_competence": mc})
        model_dirs = os.path.join(root_path, "models", f"market_competence_{mc:.1f}")
        os.makedirs(model_dirs, exist_ok=True)
                    
        envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(),
                                              amm=AMM(),
                                              market_competence=mc)) for _ in range(10)]
        env = SubprocVecEnv(envs)
        model = PPO("MlpPolicy", env=env, learning_rate=LEARNING_RATE, gamma=0.95)
        
        checkpoint_callback = CheckpointCallback(save_freq=EVALUATE_PER_STEP, save_path=model_dirs, name_prefix="rl_model")
        eval_env = Monitor(MultiAgentAmm(market=MarketSimulator(), amm=AMM(), market_competence=mc))
        eval_callback = EvalCallback(eval_env, best_model_save_path=model_dirs, log_path=model_dirs, eval_freq=EVALUATE_PER_STEP)
        wandb_callback = WandbCallback()
        
        model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, eval_callback, wandb_callback], progress_bar=True)

        wandb.finish()

if __name__ == '__main__':
    ROOT_DIR = '/Users/haofu/AMM-Python/stable_baseline/multiple_agent'
    train(ROOT_DIR)
