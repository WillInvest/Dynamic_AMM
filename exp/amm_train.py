import os
import csv
import wandb
import argparse
import sys  
import socket
import numpy as np
# Get the path to the Dynamic_AMM directory
sys.path.append(f'{os.path.expanduser("~")}/Dynamic_AMM')
# env related
from env.market import MarketSimulator
from env.new_amm import AMM
# from env.amm_env import DynamicAMM
from env.dummy_AMM import DummyAMM
from callbacks import EvalCallback, CheckpointCallback, NoiseCallback, MakerWandbCallback
    
# stable baseline related
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

def train(root_path, model_idx=None):
    n_envs = 32
    TOTAL_STEPS = n_envs * int(1e6)
    EVALUATE_PER_STEP = int(1e3)
    CHECKPOINT_PER_STEP = int(1e3)
    model_dirs = os.path.join(root_path, "dummy_maker_model")
    os.makedirs(model_dirs, exist_ok=True)
    # trader_dirs = os.path.join(root_path, "trader_model")

    wandb.init(project=f"Dynamic_AMM_Maker",
               entity='willinvest',
               name=f'User-{socket.gethostname()}_Dummy')
    
    envs = [lambda: Monitor(DummyAMM(market=MarketSimulator(),
                                            amm=AMM())) for seed in range(n_envs)]
    env = SubprocVecEnv(envs)
    # Define action noise
    n_actions = env.action_space.shape[-1]
    exploration_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))
    if model_idx is None:
        model = TD3("MlpPolicy", env, verbose=0, learning_rate=0.0003, train_freq=(n_envs*100, 'step'), gradient_steps=-1, action_noise=exploration_noise)
    else:
        model_path = f'/home/shiftpub/Dynamic_AMM/models/maker_model/rl_maker_{model_idx}_steps.zip'
        model = TD3.load(path=model_path, env=env)
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_PER_STEP, save_path=model_dirs, name_prefix="rl_maker")
    wandb_callback = MakerWandbCallback()
    eval_callback = EvalCallback(env,
                                 best_model_save_path=model_dirs,
                                 log_path=model_dirs,
                                 eval_freq=EVALUATE_PER_STEP,
                                 n_eval_episodes=n_envs,
                                 deterministic=True)
    noise_schedule_callback = NoiseCallback(start_noise=1.0, end_noise=0.1, total_steps=TOTAL_STEPS)
    model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_callback, eval_callback, wandb_callback, noise_schedule_callback], progress_bar=True)

    wandb.finish()

if __name__ == '__main__':
    
    ROOT_DIR = f'{os.path.expanduser("~")}/Dynamic_AMM/models'
    train(ROOT_DIR)
