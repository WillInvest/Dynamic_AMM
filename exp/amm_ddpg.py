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

sys.path.append("..")

from env.amm_env import ArbitrageEnv
from env.market import GBMPriceSimulator
from env.new_amm import AMM
import matplotlib.pyplot as plt

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
        self.batch_size = int(64)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(512)  # collect horizon_len step while exploring, then update network
        self.buffer_size = int(51200)  # ReplayBuffer size. First in first out for off-policy.
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
        self.eval_per_step = int(1024)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)


class Actor(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise

    def forward(self, state: Tensor) -> Tensor:
        action = self.net(state)
        return action.tanh()

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        action_avg = self.net(state).tanh()
        dist = Normal(action_avg, self.explore_noise_std)
        action = dist.sample()
        return action.clip(-1.0, 1.0)


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
        print(f"device: {self.device}")
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
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
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.act.explore_noise_std = getattr(args, 'explore_noise', 0.1)  # set for `self.act.get_action()`

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state
        get_action = self.act.get_action
        for i in range(horizon_len):
            # print(f"array_state: {ary_state}")
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action = torch.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0)).squeeze(0)

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, truncated, _ = env.step(ary_action)
            if i == horizon_len - 1:
                done = True
            if done:
                ary_state, _ = env.reset()
                
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)         
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer) -> [float]:
        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        assert update_times > 0
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            action = self.act(state)
            obj_actor = self.cri_target(state, action).mean()
            self.optimizer_update(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
            obj_actors += obj_actor.item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            states, actions, rewards, undones, next_states = buffer.sample(batch_size)
            next_actions = self.act_target(next_states)
            next_q_values = self.cri_target(next_states, next_actions)
            q_labels = rewards + undones * self.gamma * next_q_values
        q_values = self.cri(states, actions)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, states


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

def train_agent(args: Config, USING_USD):
    args.init_before_training()
    gpu_id = 0
    market = GBMPriceSimulator(start_price=args.mkt_start, deterministic=False)
    fee_rate = args.fee_rate
    amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)
    env = ArbitrageEnv(market, amm, USD=USING_USD)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.last_state, _ = env.reset()
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size,
                          state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim, )
    buffer_items = agent.explore_env(env, args.horizon_len * 10, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer
    eval_market = GBMPriceSimulator(start_price=args.mkt_start, deterministic=True)
    eval_amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)
    eval_env = ArbitrageEnv(eval_market, eval_amm, USD=USING_USD)
    evaluator = Evaluator(eval_env=eval_env,
                          eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd, fee=fee_rate, epsilon=args.epsilon, USD=USING_USD)
    torch.set_grad_enabled(False)
    while True:  # start training
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`

def save_model(actor, directory, step_count):
    directory = f'./saved_model/{directory}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"actor_step_{step_count}.pth")
    torch.save(actor.state_dict(), filepath)
    # print(f"Model saved to {filepath}")
    
class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.', fee: float = 0.0, epsilon: float = 0.002, USD=False):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps
        self.fee = fee
        self.epsilon = epsilon
        self.USD = USD
        self.best_reward = 0.0

        self.recorder = []
        print("\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'avgD':>8}  {'stdD':>6}   | {'objC':>8}  {'objA':>8}")

    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        fee_rate = self.fee
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        results = []
        distances = []
        for _ in range(self.eval_times):
            rewards, steps, distance = get_rewards_and_steps(self.env_eval, actor, fee_rate, self.epsilon, USD=self.USD)
            # Extract the first element from the rewards array
            results.append([rewards[0], steps])
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

        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {avg_d:8.2f}  {std_d:6.2f}    "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")
        
        # Save the model
        save_directory = os.path.join(self.cwd, f"model_saves_step_{self.total_step}")
        if avg_r > self.best_reward:
            save_model(actor, save_directory, self.total_step)
            self.best_reward = avg_r
            
        
        
def calculate_distance(amm_bid, amm_ask, market_bid, market_ask):
    if amm_bid > market_ask:
        # Non-overlapping: AMM higher than market
        distance = amm_bid - market_ask
    elif amm_ask < market_bid:
        # Non-overlapping: AMM lower than market
        distance = market_bid - amm_ask
    else:
        # Overlapping
        if amm_ask < market_ask:
            distance = amm_ask - market_bid
        else:
            distance = market_ask - amm_bid
            
    return distance

def plot_amm_market(amm_bid_step, amm_ask_step, market_bid, market_ask):
     # Plotting the prices
    steps = len(amm_ask_step)
    plt.figure(figsize=(10, 10))
    plt.plot(market_ask, label='Market Ask', color='red')
    plt.plot(market_bid, label='Market Bid', color='blue')
    plt.step(np.arange(steps), amm_ask_step, where='mid', label='AMM Ask', linestyle='--', color='red')
    plt.step(np.arange(steps), amm_bid_step, where='mid', label='AMM Bid', linestyle='--', color='blue')
    plt.title('Bid and Ask Ratios')
    plt.xlabel('Step')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.pause(0.1)  # Display the plot for 0.1 seconds
    plt.close()  # Close the plot window

def get_rewards_and_steps(env, actor, if_render: bool = False, fee_rate = 0.0, epsilon = 0.002, USD=True) -> (float, int):  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, _ = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    distances = []
    amm_bids = []
    amm_asks = []
    market_bids = []
    market_asks = []
    
    for episode_steps in range(50):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, truncated, _ = env.step(action)
        cumulative_returns += reward
        if USD:
            ammMid = state[1] #/ state[0]
            ammBid = ammMid / (1 + fee_rate)
            ammAsk = ammMid * (1 + fee_rate)
            amm_bids.append(ammBid)
            amm_asks.append(ammAsk)
            # askA = state[2] * (1 + 2 * epsilon)
            # askB = state[3] * (1 + epsilon)
            # bidA = state[2] / (1 + 2 * epsilon)
            # bidB = state[3] / (1 + epsilon)
            # ask_ratio = askA / bidB
            # bid_ratio = bidA / askB
            ask_ratio = state[0] * (1+epsilon)
            bid_ratio = state[0] / (1+epsilon)
            market_asks.append(ask_ratio)
            market_bids.append(bid_ratio)

        else:
            ammAsk = state[1] * (1+fee_rate)
            ammBid = state[1] / (1+fee_rate)
            amm_bids.append(ammBid)
            amm_asks.append(ammAsk)
            ask_ratio = state[0] * (1+epsilon)
            bid_ratio = state[0] / (1+epsilon)
            market_asks.append(ask_ratio)
            market_bids.append(bid_ratio)
            
        distances.append(calculate_distance(amm_ask=ammAsk, amm_bid=ammBid, market_ask=ask_ratio, market_bid=bid_ratio))
        if if_render:
            env.render()
        if done:
            break
    # plot_amm_market(amm_bid_step, amm_ask_step, market_bids, market_asks)
    return cumulative_returns, episode_steps + 1, sum(distances)


def generate_timestamp():
    # This function returns a formatted timestamp string
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

def train_ddpg_for_amm(fee_rate, gpu_id=0):
    time_stamp = generate_timestamp()
    env_args = {
        'env_name': f'AMM_{time_stamp}',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 2,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }  # env_args = get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)

    args = Config(agent_class=AgentDDPG, env_class=ArbitrageEnv, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(1e8)  # break training if 'total_step > break_step'
    args.net_dims = (32, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.95 # discount factor of future rewards
    args.fee_rate = fee_rate
    args.epsilon = 0.005
    args.mkt_start = 1

    train_agent(args, USING_USD=False)
    

if __name__ == "__main__":
    train_ddpg_for_amm(0.02)