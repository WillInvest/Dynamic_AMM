import os
import sys
import time
import wandb
import numpy as np
import warnings
from collections import deque
import random

# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress TensorFlow warnings and informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# sys.path.append('../../..')
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
from wandb.integration.sb3 import WandbCallback


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

def print_results(results=None, init=True):
    if init:
        print("\n| `total steps`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Std value of cumulative rewards, which is the std of rewards in an episode."
              "\n| `bestR`: Best reward in all episodes."
              "\n| `bestR_step`: Totals steps that generate the best reward."
              "\n| `mean_total_reward`: Mean total reward in all episodes."
              "\n| `mean_swap_rate`: Mean swap rate in all episodes."
              "\n| `mean_tip_rate`: Mean tip rate in all episodes."
              "\n| `mean_total_gas`: Mean total gas in all episodes."

              f"\n| {'step':>15}  " # | {'avgR':>15} | {'stdR':>15} |
              f"{'bestR':>15} | {'bestR_step':>15} | "
              f"{'mean_total_reward':>18} | {'mean_reward1':>18} | {'mean_reward2':>18} | {'mean_swap_rate':>15} | "
              f"{'mean_tip_rate':>15} | {'mean_total_gas':>15}")
    else:
        print(f"| {results['total_steps']:15.2e} | "
            #   f"{results['average_reward']:15.2f} | "
            #   f"{results['std_reward']:15.2f} | "
              f"{results['best_reward']:15.2f} | "
              f"{results['best_reward_step']:15.2f} | "
              f"{results['mean_total_rewards']:18.2f} | "
              f"{results['mean_reward1']:15.6f} | "
              f"{results['mean_reward2']:15.6f} | "
              f"{results['mean_swap_rate']:15.6f} | "
              f"{results['mean_tip_rate']:15.6f} | "
              f"{results['mean_total_gas']:15.6f}")


# Function to save a model to the queue
def save_model_to_queue(model, model_path, best_model_queue):
    model.save(model_path)
    best_model_queue.append(model_path)

def train(fee_rates, sigmas):
    
    TOTAL_STEPS = 5e6
    EVALUATE_PER_STEP = 1e3
    MODEL = 'TD3'
    LEARNING_RATE = 0.0001
    SEED_LEN = 5
    eval_deterministic = False
    eval_steps = 500
    exchange_steps = 30
    # Initialize the deque with a fixed size
    # best_model_queue = deque(maxlen=5)

    for risk_aversion in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for fee_rate in fee_rates:
            for sigma in sigmas:
                    fee_rate = round(fee_rate, 2)
                    sigma = round(sigma, 1)
                    config = {
                        "fee_rate": fee_rate,
                        "sigma": sigma,
                        "risk_aversion": risk_aversion,
                        "eval_deterministic": eval_deterministic,
                        "eval_steps": eval_steps
                    }
                    wandb.init(project=f'AMM_Multi_Agent_rule_based_risk_aversion_{risk_aversion}',
                            config=config,
                            name=f"f{fee_rate}-s{sigma}-rule-based")

                    ROOT_DIR = '/home/shiftpub/AMM-Python/stable_baseline/multiple_agent'
                    model_dirs = os.path.join(ROOT_DIR,
                                            "models",
                                            f'r{fee_rate:.2f}_s{sigma:.1f}'.format(fee_rate, sigma),)
                    
                    logdir = os.path.join(ROOT_DIR, "logs", "tensorboard")
                    if not os.path.exists(model_dirs):
                        os.makedirs(model_dirs)
                    if not os.path.exists(logdir):
                        os.makedirs(logdir)
                        
                    # Create vectorized environment
                    # random selected model from deque
                    # if best_model_queue:
                    #     fix_model = random.choice(best_model_queue)
                    # else:
                    #     fix_model = None

                    envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(sigma=sigma),
                                                        amm=AMM(fee=fee_rate),
                                                        rule_based=True, risk_aversion=risk_aversion)) for _ in range(10)]
                    env = SubprocVecEnv(envs)
                    n_actions = env.action_space.shape[-1]
                    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                    
                    # Create the model with specified hyperparameters
                    # if best_model_queue:
                    #     learning_model = random.choice(best_model_queue) # Randomly select a model from the queue
                    #     model = TD3.load(learning_model, env=env, learning_rate=LEARNING_RATE, action_noise=action_noise)
                    # else:
                    model = TD3("MlpPolicy", env=env, learning_rate=LEARNING_RATE, action_noise=action_noise, gamma=0.95)
                    
                    # Set up the logger
                    unique_logdir = os.path.join(logdir, f"run_{int(time.time())}")
                    new_logger = configure(unique_logdir, ["tensorboard"])
                    model.set_logger(new_logger)

                    # Initialize best reward and step count
                    best_avg_reward = -float('inf')
                    total_steps_trained = 0
                    best_reward_steps = 0
                    print_results(results=None, init=True)

                    # Training loop with evaluation
                    for _ in range(int(TOTAL_STEPS // EVALUATE_PER_STEP)):  # Adjust the range based on your needs
                        model.learn(total_timesteps=int(EVALUATE_PER_STEP), progress_bar=False, reset_num_timesteps=False)
                        total_steps_trained += int(EVALUATE_PER_STEP)
                        eval_episodes = 10
                        mean_reward, std_reward, mean_tot_rew, mean_rew1, mean_rew2, mean_swap_rate, mean_tip_rate, mean_total_gas = evaluate_policy(
                            model, env, n_eval_episodes=eval_episodes, deterministic=True)
                        model_name_best = os.path.join(model_dirs, f"{MODEL}_best_model_r_{fee_rate:.2f}_s_{sigma:.1f}_rule_based".format(fee_rate, sigma))
                        model_name_step = os.path.join(model_dirs, f"{MODEL}_step_model_r_{fee_rate:.2f}_s_{sigma:.1f}_rule_based_steps{total_steps_trained:.0f}".format(fee_rate, sigma, total_steps_trained))
                        if mean_reward >= best_avg_reward:
                            best_avg_reward = mean_reward
                            best_reward_steps = total_steps_trained
                            # save model, and update the best model queue
                            model.save(model_name_best)
                            # best_model_queue.append(model_name)
                        if total_steps_trained % 100000 == 0:
                            model.save(model_name_step)
                        
                        # Log wandb
                        wandb.log({
                            "mean_reward": mean_reward,
                            "std_reward": std_reward,
                            "best_avg_rew": best_avg_reward,
                            "best_rew_steps": best_reward_steps,
                            "total_steps_trained": total_steps_trained,
                            "mean_total_rewards": mean_tot_rew,
                            "mean_reward1": mean_rew1,
                            "mean_reward2": mean_rew2,
                            "mean_swap_rate": mean_swap_rate,
                            "mean_tip_rate": mean_tip_rate,
                            "mean_total_gas": mean_total_gas
                            })

                        results = {
                            "total_steps": total_steps_trained,
                            # "average_reward": mean_reward,
                            # "std_reward": std_reward,
                            "best_reward": best_avg_reward,
                            "best_reward_step": best_reward_steps,
                            "mean_total_rewards": mean_tot_rew,
                            "mean_reward1": mean_rew1,
                            "mean_reward2": mean_rew2,
                            "mean_swap_rate": mean_swap_rate,
                            "mean_tip_rate": mean_tip_rate,
                            "mean_total_gas": mean_total_gas
                        }
                        print_results(results=results, init=False)
                        
                    # Finish the wandb run
                    wandb.finish()

if __name__ == "__main__":
    
    rates = np.arange(0.05, 0.21, 0.01)
    sigmas = np.arange(0.2, 1.0, 0.2)
    train(fee_rates=rates, sigmas=sigmas)

