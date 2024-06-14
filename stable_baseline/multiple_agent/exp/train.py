import os
import sys
import time
import wandb
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress TensorFlow warnings and informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.append('../../..')

from env.multiAmm import MultiAgentAmm
from env.market import MarketSimulator
from env.new_amm import AMM

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
              f"\n| {'step':>15} | {'avgR':>15} | {'stdR':>15} | {'bestR':>15} | {'bestR_step':>15}")
    else:
        print(f"| {results['total_steps']:15.2e} | "
              f"{results['average_reward']:15.2f} | "
              f"{results['std_reward']:15.2f} | "
              f"{results['best_reward']:15.2f} | "
              f"{results['best_reward_step']:15.2f}")

def train(fee_rates, sigmas):
    
    TOTAL_STEPS = 1e6
    EVALUATE_PER_STEP = 1e3
    MODEL = 'TD3'
    LEARNING_RATE = 0.001
    SEED_LEN = 5
    eval_deterministic = False
    eval_steps = 500
    exchange_steps = 10
    
    for agent_seed in range(SEED_LEN):
        for fee_rate in fee_rates:
            for sigma in sigmas:
                for ex in range(1, exchange_steps):
                    fee_rate = round(fee_rate, 2)
                    sigma = round(sigma, 2)
                    config = {
                        "fee_rate": fee_rate,
                        "sigma": sigma,
                        "agent_seed": agent_seed,
                        "eval_deterministic": eval_deterministic,
                        "eval_steps": eval_steps
                    }
                    wandb.init(project=f'AMM_Multi_Agent_{MODEL}',
                            config=config,
                            name=f"{MODEL}{agent_seed}-f{fee_rate}-s{sigma}-exstep{ex}")

                    ROOT_DIR = '/home/shiftpub/AMM-Python/stable_baseline'
                    model_dirs = os.path.join(ROOT_DIR,
                                            "models",
                                            MODEL,
                                            f'agent_seed_{agent_seed}',
                                            f'fee_{fee_rate:.2f}',
                                            f'sigma_{sigma:.1f}')
                    
                    logdir = os.path.join(ROOT_DIR, "logs", "tensorboard")
                    if not os.path.exists(model_dirs):
                        os.makedirs(model_dirs)
                    if not os.path.exists(logdir):
                        os.makedirs(logdir)
                        
                    # Create vectorized environment
                    model_name = os.path.join(model_dirs, f"{MODEL}_best_model")
                    envs = [lambda: Monitor(MultiAgentAmm(market=MarketSimulator(sigma=sigma),
                                                        amm=AMM(fee=fee_rate),
                                                        model_path=model_name), filename=None) for _ in range(10)]
                    env = SubprocVecEnv(envs)
                    n_actions = env.action_space.shape[-1]
                    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
                    
                    # Create the model with specified hyperparameters
                    model = TD3(policy='MlpPolicy',
                                env=env,
                                learning_rate=LEARNING_RATE,
                                buffer_size=int(1e6),
                                gamma=0.95,
                                action_noise=action_noise,
                                seed=int(agent_seed))
                    
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
                        mean_reward, std_reward, mean_total_rewards = evaluate_policy(model, env, n_eval_episodes=eval_episodes, deterministic=True)
                        
                        if mean_reward >= best_avg_reward:
                            best_avg_reward = mean_reward
                            best_reward_steps = total_steps_trained
                            model.save(model_name)
                            
                        # Log wandb
                        wandb.log({
                            "mean_reward": mean_reward,
                            "std_reward": std_reward,
                            "best_avg_rew": best_avg_reward,
                            "best_rew_steps": best_reward_steps,
                            "total_steps_trained": total_steps_trained
                            })

                        results = {
                            "total_steps": total_steps_trained,
                            "average_reward": mean_reward,
                            "std_reward": std_reward,
                            "best_reward": best_avg_reward,
                            "best_reward_step": best_reward_steps
                        }
                        print_results(results=results, init=False)
                        
                    # Finish the wandb run
                    wandb.finish()

if __name__ == "__main__":
    
    rates = np.arange(0.01, 0.21, 0.01)
    sigmas = np.arange(0.2, 1.2, 0.2)
    train(fee_rates=rates, sigmas=sigmas)

