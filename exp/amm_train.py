import os
import csv
import wandb
import argparse
import sys  
import socket
import numpy as np
# Get the path to the AMM-Python directory
sys.path.append(f'{os.path.expanduser("~")}/AMM-Python')
# env related
from env.market import MarketSimulator
from env.new_amm import AMM
from env.amm_env import DynamicAMM
    
# stable baseline related
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
class WandbCallback(BaseCallback):
    def __init__(self, log_name='trade_log.csv', verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.log_file = log_name
        self.log_headers_written = False
        
        # Check if the log file exists and delete it
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _on_step(self) -> bool:
        cumulative_fee = self.locals['infos'][0]['cumulative_fee']
        cumulative_pnl = self.locals['infos'][0]['cumulative_pnl']
        total_pnl = self.locals['infos'][0]['total_pnl']
        total_fee = self.locals['infos'][0]['total_fee']
        actions = self.locals['actions'][0]
        done = self.locals['dones'][0]
        swap_rates = self.locals['infos'][0]['swap_rates']
        urgent_levels = self.locals['infos'][0]['urgent_levels']
        reward = self.locals['rewards'][0]

        # Prepare log data
        log_data = {
            "actions": actions,
            'reward': reward
        }
        
        # Add urgent_levels with a prefix
        for i, level in enumerate(urgent_levels):
            log_data[f"urgent_levels/urgent_level_{i}"] = level
            
        if done:
            wandb.log({
                "cumulative_fee": cumulative_fee,
                "cumulative_pnl": cumulative_pnl
            })    
            
            for trader_id in total_pnl.keys():
                wandb.log({
                    f"total_pnl/trader_{trader_id}": total_pnl[trader_id],
                    f"total_fee/trader_{trader_id}": total_fee[trader_id]
                })
        
        # Log the data to wandb
        wandb.log(log_data)
        for trader_id in swap_rates.keys():
            wandb.log({f"swap_rate/trader_{trader_id}": swap_rates[trader_id]})

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
    EVALUATE_PER_STEP = int(1e2)
    CHECKPOINT_PER_STEP = int(5e3)
    model_dirs = os.path.join(root_path, "maker_model")
    os.makedirs(model_dirs, exist_ok=True)
    trader_dirs = os.path.join(root_path, "trader_model")

    wandb.init(project=f"Dynamic_AMM_Maker",
               entity='willinvest',
               name=f'User-{socket.gethostname()}')
    
    n_envs = 16
    envs = [lambda: Monitor(DynamicAMM(market=MarketSimulator(),
                                            amm=AMM(),
                                            trader_dir=trader_dirs)) for seed in range(n_envs)]
    env = SubprocVecEnv(envs)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, verbose=1, learning_rate=0.0003, train_freq=(1, 'step'), gradient_steps=-1, action_noise=action_noise)
        
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_PER_STEP, save_path=model_dirs, name_prefix="rl_maker")
    wandb_callback = WandbCallback()
    eval_callback = EvalCallback(env,
                                 best_model_save_path=model_dirs,
                                 log_path=model_dirs,
                                 eval_freq=EVALUATE_PER_STEP,
                                 n_eval_episodes=1,
                                 deterministic=True,
                                 render=False)
        
    model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, wandb_callback], progress_bar=True)

    wandb.finish()

if __name__ == '__main__':
    
    ROOT_DIR = f'{os.path.expanduser("~")}/AMM-Python/models'
    train(ROOT_DIR)
