import os
import sys
import time
import numpy as np
import warnings
import tensorflow as tf
from collections import deque
import random
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress TensorFlow warnings and informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append('../../..')
print(f"current working directory: {os.getcwd()}")
print(f"sys path: {sys.path}")

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

def create_model(model_type, env, learning_rate, buffer_size, batch_size, tau, gamma, action_noise):
    """Create the model with specified hyperparameters."""
    if model_type == "TD3":
        return TD3("MlpPolicy", 
                   env, 
                   action_noise=action_noise,
                   verbose=1, 
                   gamma=gamma, 
                   learning_rate=learning_rate, 
                   buffer_size=buffer_size, 
                   batch_size=batch_size, 
                   tau=tau)
        
def train(fee_rates, sigmas):
    
    TOTAL_STEPS = 1e6
    EVALUATE_PER_STEP = 1e4
    MODEL = 'TD3'
    LEARNING_RATE = 0.0001
    SEED_LEN = 5
    eval_deterministic = False
    eval_steps = 500
    exchange_steps = 30
    log_dir = "logs/"
    
    for risk_aversion in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for fee_rate in fee_rates:
            for sigma in sigmas:
                sigma = round(sigma, 2)
                ROOT_DIR = '/home/shiftpub/AMM-Python/stable_baseline/multiple_agent'
                model_dirs = os.path.join(ROOT_DIR,
                                          "models", f"risk_aversion_{risk_aversion}",
                                          f'r{fee_rate:.4f}_s{sigma:.2f}')
                
                if not os.path.exists(model_dirs):
                    os.makedirs(model_dirs)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    
                envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(sigma=sigma),
                                                      amm=AMM(fee=fee_rate),
                                                      rule_based=True, risk_aversion=risk_aversion)) for _ in range(10)]
                env = SubprocVecEnv(envs)
                n_actions = env.action_space.shape[-1]
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                
                model = TD3("MlpPolicy", env=env, learning_rate=LEARNING_RATE, action_noise=action_noise, gamma=0.95)
                
                # Set up the logger
                unique_logdir = os.path.join(log_dir, f"run_{int(time.time())}")
                summary_writer = tf.summary.create_file_writer(unique_logdir)

                # Initialize best reward and step count
                best_avg_reward = -float('inf')
                total_steps_trained = 0
                best_reward_steps = 0
                
                # Training loop with evaluation
                for _ in tqdm(range(int(TOTAL_STEPS // EVALUATE_PER_STEP)), f"risk_aversion: {risk_aversion}, fee_rate: {fee_rate}, sigma: {sigma}"):
                    model.learn(total_timesteps=int(EVALUATE_PER_STEP), progress_bar=False, reset_num_timesteps=False)
                    total_steps_trained += int(EVALUATE_PER_STEP)
                    eval_episodes = 10
                    mean_reward, std_reward = evaluate_policy(
                        model, env, n_eval_episodes=eval_episodes, deterministic=True)
                    model_name_best = os.path.join(model_dirs, f"{MODEL}_best_model_r_{fee_rate:.2f}_s_{sigma:.1f}_rule_based".format(fee_rate, sigma))
                    model_name_step = os.path.join(model_dirs, f"{MODEL}_step_model_r_{fee_rate:.2f}_s_{sigma:.1f}_rule_based_steps{total_steps_trained:.0f}".format(fee_rate, sigma, total_steps_trained))
                    if mean_reward >= best_avg_reward:
                        best_avg_reward = mean_reward
                        best_reward_steps = total_steps_trained
                        # save model, and update the best model queue
                        model.save(model_name_best)
                    if total_steps_trained % 100000 == 0:
                        model.save(model_name_step)
                    
                    # Log results to TensorFlow
                    with summary_writer.as_default():
                        tf.summary.scalar('Average Reward', mean_reward, step=total_steps_trained)
                        tf.summary.scalar('Standard Deviation of Reward', std_reward, step=total_steps_trained)
                        tf.summary.scalar('Total Steps', total_steps_trained, step=total_steps_trained)
                        summary_writer.flush()

if __name__ == "__main__":
    
    rates = [0.0001, 0.0005]
    # Create the initial array with increments of 0.01
    sigmas = np.arange(0.01, 0.11, 0.01)

    # Define additional sigma values
    additional_sigmas = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Concatenate the arrays
    sigmas = np.concatenate((sigmas, additional_sigmas))
    train(fee_rates=rates, sigmas=sigmas)
