import os
import sys
import time
from copy import deepcopy
import time
import gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import wandb

sys.path.append("..")

from env.uniswap_env import UniSwapEnv
from env.market import MarketSimulator
from env.new_amm import AMM
from env.gas_fee import GasFeeSimulator
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:  # for off-policy
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = True  # whether off-policy or on-policy of DRL algorithm

        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        '''Arguments for reward shaping'''
        self.gamma = 0.95  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(500)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(500)  # collect horizon_len step while exploring, then update network
        self.buffer_size = int(500000)  # ReplayBuffer size. First in first out for off-policy.
        self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        self.eval_times = int(1)  # number of times that get episodic cumulative return
        self.eval_per_step = int(1000)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'{self.env_name}_{self.agent_class.__name__[5:]}_{self.seed}'
        # os.makedirs(self.cwd, exist_ok=True)


class Actor(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        # Set default standard deviations for exploration noise for each action dimension
        self.explore_noise_std_0 = 0.1  # For the bounded action
        self.explore_noise_std_1 = 0.1  # For the unbounded action

    def forward(self, state: Tensor) -> Tensor:
        action = self.net(state)
        # Apply tanh to the first dimension to bound it between [-1, 1]
        action[:, 0] = torch.tanh(action[:, 0])
        # Apply sigmoid to the second dimension to bound it between [0, 1]
        action[:, 1] = torch.sigmoid(action[:, 1])
        return action

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        action_avg = self.forward(state)
        # Create different normal distributions for each action dimension
        dist_0 = Normal(action_avg[:, 0], self.explore_noise_std_0)
        dist_1 = Normal(action_avg[:, 1], self.explore_noise_std_1)
        # Sample from the distributions
        action_0 = dist_0.sample().clamp(-1.0, 1.0)  # Apply clipping to the first action dimension only
        action_1 = dist_1.sample()  # No clipping for the second action dimension
        # Concatenate the actions along the correct dimension
        action = torch.cat((action_0.unsqueeze(1), action_1.unsqueeze(1)), dim=1)
        return action



class Critic(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((state, action), dim=1))  # Q value


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


def get_gym_env_args(env, if_print: bool) -> dict:
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list
        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        action_dim = env.action_space.n if if_discrete else env.action_space.shape[0]
    else:
        env_name = env.env_name
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    print(f"env_args = {repr(env_args)}") if if_print else None
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.soft_update_tau = args.soft_update_tau

        self.last_state = None  # save the last state of the trajectory for training. `last_state.shape == (state_dim)`
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act1 = self.act_target1 = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri1 = self.cri_target1 = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act
        self.act2 = self.act_target2 = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri2 = self.cri_target2 = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer1 = torch.optim.Adam(self.act1.parameters(), self.learning_rate)
        self.cri_optimizer1 = torch.optim.Adam(self.cri1.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        self.act_optimizer2 = torch.optim.Adam(self.act2.parameters(), self.learning_rate)
        self.cri_optimizer2 = torch.optim.Adam(self.cri2.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        # assert target_net is not current_net
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentDDPG(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor)  # get the attribute of object `self`, set Actor in default
        self.cri_class = getattr(self, 'cri_class', Critic)  # get the attribute of object `self`, set Critic in default
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)
        self.act_target1 = deepcopy(self.act1)
        self.cri_target1 = deepcopy(self.cri1)
        self.act_target2 = deepcopy(self.act2)
        self.cri_target2 = deepcopy(self.cri2)

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions1 = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards1 = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)
        actions2 = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards2 = torch.zeros((horizon_len, 1), dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state
        # get_action1 = self.act1.get_action
        # get_action2 = self.act2.get_action

        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action1 = (torch.rand(self.action_dim) * 2 - 1.0).unsqueeze(0) if if_random else self.act1.get_action(state.unsqueeze(0))
            action2 = (torch.rand(self.action_dim) * 2 - 1.0).unsqueeze(0) if if_random else self.act1.get_action(state.unsqueeze(0))

            ary_action1 = action1.detach().cpu().numpy()[0]
            ary_action2 = action2.detach().cpu().numpy()[0]
            
            ary_state, reward1, reward2, done, truncated, _ = env.step(ary_action1, ary_action2)
            if i == horizon_len - 1:
                done = True
            if done:
                ary_state, _ = env.reset()

            # Direct insertion of rewards is fine since they are already 2D tensors
            rewards1[i] = reward1
            rewards2[i] = reward2

            states[i] = state
            actions1[i] = action1
            actions2[i] = action2
            dones[i] = done

        self.last_state = ary_state
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return [states, actions1, rewards1, undones], [states, actions2, rewards2, undones]

    def update_net(self, buffer1, buffer2) -> [float]:
        obj_critics1 = obj_actors1 = obj_critics2 = obj_actors2 = 0.0
        update_times = int(self.batch_size)
        assert update_times > 0
        for i in range(update_times):
            obj_critic1, obj_critic2, state1, state2 = self.get_obj_critic(buffer1, buffer2, self.batch_size)

            self.optimizer_update(self.cri_optimizer1, obj_critic1)
            self.optimizer_update(self.cri_optimizer2, obj_critic2)

            self.soft_update(self.cri_target1, self.cri1, self.soft_update_tau)
            self.soft_update(self.cri_target2, self.cri2, self.soft_update_tau)

            obj_critics1 += obj_critic1.item()
            obj_critics2 += obj_critic2.item()

            action1 = self.act1(state1)
            action2 = self.act2(state2)

            obj_actor1 = self.cri_target1(state1, action1).mean()
            obj_actor2 = self.cri_target2(state2, action2).mean()

            self.optimizer_update(self.act_optimizer1, -obj_actor1)
            self.optimizer_update(self.act_optimizer2, -obj_actor2)

            self.soft_update(self.act_target1, self.act1, self.soft_update_tau)
            self.soft_update(self.act_target2, self.act2, self.soft_update_tau)

            obj_actors1 += obj_actor1.item()
            obj_actors2 += obj_actor2.item()

        return [obj_critics1 / update_times, obj_actors1 / update_times], [obj_critics2 / update_times, obj_actors2 / update_times] 

    def get_obj_critic(self, buffer1, buffer2, batch_size: int):
        with torch.no_grad():
            states1, actions1, rewards1, undones1, next_states1 = buffer1.sample(batch_size)
            next_actions1 = self.act_target1(next_states1)
            next_q_values1 = self.cri_target1(next_states1, next_actions1)
            q_labels1 = rewards1 + undones1 * self.gamma * next_q_values1
            states2, actions2, rewards2, undones2, next_states2 = buffer2.sample(batch_size)
            next_actions2 = self.act_target2(next_states2)
            next_q_values2 = self.cri_target2(next_states2, next_actions2)
            q_labels2 = rewards2 + undones2 * self.gamma * next_q_values2

        q_values1 = self.cri1(states1, actions1)
        obj_critic1 = self.criterion(q_values1, q_labels1)
        q_values2 = self.cri2(states2, actions2)
        obj_critic2 = self.criterion(q_values2, q_labels2)
        
        return obj_critic1, obj_critic2, states1, states2


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [Tensor]):
        states, actions, rewards, undones = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]

def train_agent(args: Config):
    args.init_before_training()
    gpu_id = 0
    market = MarketSimulator(start_price=args.mkt_start, deterministic=False)
    fee_rate = args.fee_rate
    amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)
    gas = GasFeeSimulator()
    env = UniSwapEnv(market, amm, gas)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.last_state, _ = env.reset()
    buffer1 = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size,
                          state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim, )
    buffer2 = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size,
                          state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim, )
    buffer_items1, buffer_items2 = agent.explore_env(env, args.batch_size * 100, if_random=True)
    buffer1.update(buffer_items1)  # warm up for ReplayBuffer
    buffer2.update(buffer_items2)  # warm up for ReplayBuffer
    eval_market = MarketSimulator(start_price=args.mkt_start, deterministic=args.deterministic)
    eval_env = UniSwapEnv(eval_market, amm, gas)
    evaluator = Evaluator(eval_env=eval_env,
                          eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd, fee=fee_rate, epsilon=args.epsilon)
    torch.set_grad_enabled(False)
    while True:  # start training
        buffer_items1, buffer_items2 = agent.explore_env(env, args.horizon_len)
        buffer1.update(buffer_items1)
        buffer2.update(buffer_items2)

        torch.set_grad_enabled(True)
        logging_tuple1, logging_tuple2 = agent.update_net(buffer1, buffer2)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act1, agent.act2, args.horizon_len, logging_tuple1, logging_tuple2, saving_name=args.saving_name)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`

def save_model(actor1, actor2, directory, step_count, fee_rate, saving_name):
    directory = f'./{saving_name}/{fee_rate:.2f}/{directory}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath1 = os.path.join(directory, f"actor1.pth")
    filepath2 = os.path.join(directory, f"actor2.pth")

    torch.save(actor1.state_dict(), filepath1)
    torch.save(actor2.state_dict(), filepath2)
    # print(f"Model saved to {filepath}")
    
class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.', fee: float = 0.0, epsilon: float = 0.002):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps
        self.fee = fee
        self.epsilon = epsilon
        self.best_reward = -1e6
        self.best_reward_step = 0

        self.recorder = []
        # Initialize wandb
        wandb.init(project="AMM_Multiple_Agent_DDPG")
        # Description of the metrics
        description = (
            "\n| `total_steps`: Number of samples, or total training steps, or running times of `env.step()`."
            "\n| `average_steps`: Average step for episodes"
            "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
            "\n| `bestR`: Best reward in all episodes."
            "\n| `bestR_step`: Totals steps that generate the best reward."
            "\n| `objC1`: Objective of Critic network1. Or call it loss function of critic network1."
            "\n| `objC2`: Objective of Critic network2. Or call it loss function of critic network2."
            "\n| `objA1`: Objective of Actor network1. It is the average Q value of the critic network1."
            "\n| `objA2`: Objective of Actor network2. It is the average Q value of the critic network2."
        )

        # Print the description
        print(description)

        # Header for the data
        header = (
            f"\n| {'total_steps':>12} | {'avg_steps':>12} | {'avgR':>12} | {'bestR':>12} | {'bestR_step':>12} | "
            f"{'objC1':>12} | {'objC2':>12} | {'objA1':>12} | {'objA2':>12} |"
        )

        # Print the header
        print(header)

    def evaluate_and_save(self, actor1, actor2, horizon_len: int, logging_tuple1: tuple, logging_tuple2: tuple, saving_name):
        self.total_step += horizon_len
        fee_rate = self.fee
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        results = []
        distances = []
        for _ in range(self.eval_times):
            rewards, steps, distance = get_rewards_and_steps(self.env_eval, actor1, actor2, fee_rate, self.epsilon)
            # Extract the first element from the rewards array
            results.append([rewards, steps])
            distances.append(distance)

        results_array = np.array(results, dtype=np.float32)
        distance_array = np.array(distances, dtype=np.float32)
        rewards = results_array[:, 0]  # All rewards
        steps = results_array[:, 1]    # All steps
        
        avg_d = distance_array.mean()
        std_d = distance_array.std()

        avg_r = rewards.mean()  # Average of cumulative rewards
        std_r = rewards.std()   # Standard deviation of cumulative rewards
        avg_s = steps.mean()    # Average of steps per episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        # Data for the metrics
        data = (
            f"| {self.total_step:12.2e} | {avg_s:12.2f} | {avg_r:12.2f} | {self.best_reward:12.2f} | "
            f"{self.best_reward_step:12.2f} | {logging_tuple1[0]:12.2f} | {logging_tuple2[0]:12.2f} | "
            f"{logging_tuple1[1]:12.2f} | {logging_tuple2[1]:12.2f} |"
        )

        # Print the data
        print(data)
    
        # Log metrics to wandb
        wandb.log({
            "total_steps": self.total_step,
            "avg_steps": avg_s,
            "avgR": avg_r,
            "bestR": self.best_reward,
            "bestR_step": self.best_reward_step,
            "objC1": logging_tuple1[0],
            "objC2": logging_tuple2[0],
            "objA1": logging_tuple1[1],
            "objA2": logging_tuple2[1],
        })
        
        # Save the model
        save_directory = os.path.join(self.cwd, f"model_saves_step_{self.total_step}")
        if avg_r > self.best_reward:
            save_model(actor1, actor2, save_directory, self.total_step, fee_rate=self.fee, saving_name=saving_name)
            self.best_reward = avg_r
            self.best_reward_step = self.total_step
            

def get_rewards_and_steps(env, actor1, actor2, if_render: bool = False, fee_rate = 0.0, epsilon = 0.002) -> (float, int):  # cumulative_rewards and episode_steps
    device = next(actor1.parameters()).device  # net.parameters() is a Python generator.

    state, _ = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    distances = []
    amm_bids = []
    amm_asks = []
    market_bids = []
    market_asks = []
    
    for episode_steps in range(500):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action1 = actor1(tensor_state)
        tensor_action2 = actor2(tensor_state)
        action1 = tensor_action1.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        action2 = tensor_action2.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward1, reward2, done, truncated, _ = env.step(action1, action2)
        cumulative_returns += reward1 + reward2
        ammMid = state[1] 
        ammBid = ammMid / (1 + fee_rate)
        ammAsk = ammMid * (1 + fee_rate)
        amm_bids.append(ammBid)
        amm_asks.append(ammAsk)
        ask_ratio = state[0] * (1+epsilon)
        bid_ratio = state[0] / (1+epsilon)
        market_asks.append(ask_ratio)
        market_bids.append(bid_ratio)

        if done:
            break
    # plot_amm_market(amm_bid_step, amm_ask_step, market_bids, market_asks)
    return cumulative_returns, episode_steps + 1, sum(distances)


def generate_timestamp():
    # This function returns a formatted timestamp string
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

def train_ddpg_for_amm(fee_rate, seed, gpu_id=0):
    env_args = {
        'env_name': f'AMM',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 2,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }  # env_args = get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)

    args = Config(agent_class=AgentDDPG, env_class=UniSwapEnv, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(1e8)  # break training if 'total_step > break_step'
    args.net_dims = (256, 256)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.95 # discount factor of future rewards
    args.fee_rate = fee_rate
    args.epsilon = 0.005
    args.mkt_start = 1
    args.seed = seed
    args.saving_name = 'saved_model_multiple_agents_random'
    args.deterministic = False

    train_agent(args)
    

if __name__ == "__main__":
    # rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4]
    # rates = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23]
    # rates = np.arange(0.01, 0.24, 0.01)
    # rates = [0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
    rates = np.arange(0.01, 0.11, 0.01)
    for rate in rates:
        for seed in range(5):
            set_seed(seed=seed)
            train_ddpg_for_amm(fee_rate=rate, seed=seed)