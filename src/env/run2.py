"""
run.py

Tested with:
Gym 0.17.3

Zheng Xing <zxing@stevens.edu>
"""

from ddpg import *
import gym
from AmmEnv import ArbitrageEnv
from market import GBMPriceSimulator
from new_amm import AMM
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="DDPG-AMMArbitrage",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.0001,
        "architecture": "DDPG",
        "dataset": "AMM",
        "epochs": 100,
    }
)

wandb.log({"modification": "change the sigma of noise term to 0.1 from 0.2"})

RENDER = True
TOTAL_EPISODES = 200
MAX_STEPS_PER_EPISODE = 500
UPDATE_STEPS = 10

market = GBMPriceSimulator()
amm = AMM(initial_a=10000, initial_b=10000, fee=0.0)
env = ArbitrageEnv(market, amm)
observation_shape = env.observation_space.shape
action_shape = env.action_space.shape
agent = DeepDeterministicPolicyGradient(env, wandb)
print("\n\nobservation_space.shape = ", observation_shape)
print("action_space.shape = ", action_shape)

np.set_printoptions(precision=4)

for episode_cnt in range(TOTAL_EPISODES):
    state = agent.reset()
    reward_list = []
    step_cnt = 0
    while step_cnt < MAX_STEPS_PER_EPISODE:
        step_cnt += 1

        action = agent.policy(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward_list.append(reward)
        if SAVING_RAM:
            agent.memory.add((state, action, reward, done))
        else:
            agent.memory.add((state, action, reward, next_state, done))
        state = np.copy(next_state)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = agent.memory.sample()
        agent.update_online_network(state_batch, action_batch, reward_batch, next_state_batch)
        agent.update_target_network()

        if done or step_cnt == MAX_STEPS_PER_EPISODE - 1:
            episode_return = sum(reward_list)
            wandb.log({"Episode": episode_cnt, "Episode Return": episode_return})
            print("\nEpisode: #", episode_cnt, " Return: ", episode_return)
            break

