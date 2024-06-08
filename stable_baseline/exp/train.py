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

# Set up wandb
wandb.login(key='df587b8f37d917e537efda9f9c04f8155b6d49f1')
sys.path.append('../..')

from env.amm_env import ArbitrageEnv
from env.market import GBMPriceSimulator
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
              "\n| `bestR`: Best reward in all episodes."
              "\n| `bestR_step`: Totals steps that generate the best reward."
              f"\n| {'step':>15} | {'avgR':>15} | {'bestR':>15} | {'bestR_step':>15}")
    else:
        print(f"| {results['total_steps']:15.2e} | "
              f"{results['average_reward']:15.2f} | "
              f"{results['best_reward']:15.2f} | "
              f"{results['best_reward_step']:15.2f}")

if __name__ == "__main__":
    
    TOTAL_STEPS = 5e6
    EVALUATE_PER_STEP = 1e4
    MODEL = 'TD3'
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    ROOT_DIR = '/home/shiftpub/AMM-Python/stable_baseline'
    model_dirs = os.path.join(ROOT_DIR, "models", MODEL, timestamp)
    logdir = os.path.join(ROOT_DIR, "logs", "tensorboard")

    if not os.path.exists(model_dirs):
        os.makedirs(model_dirs)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    # Initialize Weights & Biases
    wandb.init(
        project="AMM-SB3", 
        config={"total_timesteps": TOTAL_STEPS, "env_name": f"ArbitrageEnv-{MODEL}"},
        sync_tensorboard=True  # ensures tensorboard and wandb logs are in sync
    )
    
    # Create the Wandb callback
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=os.path.join(wandb.run.dir, "models"),
        verbose=2,
    )
    
    # Fetch hyperparameters from wandb config
    config = wandb.config
    learning_rate = config.learning_rate
    # buffer_size = config.buffer_size
    # batch_size = config.batch_size
    # tau = config.tau
    # gamma = config.gamma

    # Create vectorized environment
    envs = [lambda: Monitor(ArbitrageEnv(market=GBMPriceSimulator(), amm=AMM()), filename=None) for _ in range(10)]
    env = SubprocVecEnv(envs)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Create the model with specified hyperparameters
    # model = create_model(model_type=MODEL,
    #                      env=env,
    #                      learning_rate=learning_rate,
    #                      buffer_size=buffer_size, 
    #                      batch_size=batch_size, 
    #                      tau=tau, 
    #                      gamma=gamma,
    #                      action_noise=action_noise)
    model = TD3(policy='MlpPolicy',
                env=env,
                verbose=0,
                learning_rate=learning_rate)
                # buffer_size=buffer_size,
                # batch_size=batch_size,
                # action_noise=action_noise,
                # tau=tau,
                # gamma=gamma)

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
        eval_deterministic = True
        model.learn(total_timesteps=int(EVALUATE_PER_STEP), progress_bar=False, reset_num_timesteps=False, callback=wandb_callback)
        total_steps_trained += int(EVALUATE_PER_STEP)
        model_filename = os.path.join(model_dirs, f"{MODEL}_{int(total_steps_trained)}")
        eval_env = Monitor(ArbitrageEnv(market=GBMPriceSimulator(deterministic=eval_deterministic), amm=AMM()))
        eval_episodes = 1 if eval_deterministic else 10
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
        
        # Log mean_reward and std_reward to wandb
        wandb.log({"mean_reward": mean_reward, "std_reward": std_reward, "total_steps_trained": total_steps_trained})
        if mean_reward >= best_avg_reward:
            best_avg_reward = mean_reward
            best_reward_steps = total_steps_trained
            model.save(model_filename)
        
        results = {
            "total_steps": total_steps_trained,
            "average_reward": mean_reward,
            "best_reward": best_avg_reward,
            "best_reward_step": best_reward_steps
        }
        print_results(results=results, init=False)
        
    # Finish the wandb run
    wandb.finish()
