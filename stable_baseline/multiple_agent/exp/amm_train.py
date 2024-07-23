import os
import sys
sys.path.append('../../..')

import time
import numpy as np
import warnings
import tensorflow as tf
from collections import deque
import random
import csv
from tqdm import tqdm
import wandb

from stable_baseline.env.amm_env import DynamicAMM
from stable_baseline.env.market import MarketSimulator
from stable_baseline.env.new_amm import AMM

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, log_name='trade_log.csv', verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.log_file = log_name
        self.log_headers_written = False
        
        # Check if the log file exists and delete it
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _on_step(self) -> bool:
        # Get cumulative fees from all sub-environments
        cumulative_fee = self.training_env.get_attr('cumulative_fee')[0]
        step_count = self.training_env.get_attr('step_count')[0]
        fee_rate = self.training_env.get_attr('fee_rate')[0]
        action = self.training_env.get_attr('action')[0]
        
        # Convert to native Python types
        cumulative_fee = float(cumulative_fee)
        step_count = int(step_count)
        fee_rate = float(fee_rate)
        action = float(action)
        
        # Log data including flattened details
        log_data = {
            "step_count": step_count,
            "fee": cumulative_fee,
            "fee_rate": fee_rate,
            "action": action
        }
        self.log_to_csv(log_data)
        
        wandb.log({
            "fee": cumulative_fee,
            "step_count": step_count,
            "fee_rate": fee_rate,
            "action": action
        })
        return True

    def log_to_csv(self, log_data):
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
            if not file_exists or not self.log_headers_written:
                writer.writeheader()
                self.log_headers_written = True
            writer.writerow(log_data)

def train(root_path):
    
    TOTAL_STEPS = int(1e7)
    EVALUATE_PER_STEP = int(1e4)
    LEARNING_RATE = 0.0001
    model_dirs = os.path.join(root_path, "models", "market_maker")
    trader_dirs = os.path.join(root_path, "models")


    wandb.init(project="AMM_RL_Market_Maker", entity='willinvest')
                    
    envs = [lambda: Monitor(DynamicAMM(market=MarketSimulator(),
                                            amm=AMM(),
                                            trader_dir=trader_dirs)) for _ in range(10)]
    env = SubprocVecEnv(envs)
    model = PPO("MlpPolicy", env=env, learning_rate=LEARNING_RATE, gamma=0.95, n_steps=4096, n_epochs=10)
        
    checkpoint_callback = CheckpointCallback(save_freq=EVALUATE_PER_STEP, save_path=model_dirs, name_prefix="rl_maker")
    # eval_env = Monitor(MultiAgentAmm(market=MarketSimulator(), amm=AMM(), market_competence=mc))
    # eval_callback = EvalCallback(eval_env, best_model_save_path=model_dirs, log_path=model_dirs, eval_freq=EVALUATE_PER_STEP)
    wandb_callback = WandbCallback()
        
    model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, wandb_callback], progress_bar=True)

    wandb.finish()

if __name__ == '__main__':
    ROOT_DIR = '/home/shiftpub/AMM-Python/stable_baseline/multiple_agent'
    train(ROOT_DIR)
