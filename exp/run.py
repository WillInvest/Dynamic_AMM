
import argparse
import datetime
import os
import sys
import pprint
import numpy as np
import torch
# Add the parent directory to the system path
sys.path.append('..')


from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, PrioritizedVectorReplayBuffer, Batch
from tianshou.env.venvs import DummyVectorEnv, SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.highlevel.logger import LoggerFactoryDefault

from env.amm_env import ArbitrageEnv
from env.market import GBMPriceSimulator
from env.new_amm import AMM

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="AMM")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--buffer-size", type=int, default=1e6)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.0005)
    parser.add_argument("--exploration-noise", type=float, default=0.01)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=50)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--USING_USD", type=bool, default=True)
    parser.add_argument("--mkt_start", type=float, default=1.0)
    parser.add_argument("--fee_rate", type=float, default=0.02)

    return parser.parse_args()

def create_env(market, amm, USD):
    """Function to create and return a new environment."""
    def _env():
        return ArbitrageEnv(market, amm, USD=USD)
    return _env

def test_ddpg(fee_rate, args: argparse.Namespace = get_args()) -> None:
    args.fee_rate = fee_rate
    market = GBMPriceSimulator(start_price=args.mkt_start, deterministic=False)
    fee_rate = args.fee_rate
    amm = AMM(initial_a=10000, initial_b=10000, fee=fee_rate)
    env = ArbitrageEnv(market, amm, USD=args.USING_USD)
    eval_market = GBMPriceSimulator(start_price=args.mkt_start, deterministic=True)
    # test_env = ArbitrageEnv(eval_market, amm, USD=args.USING_USD)
    train_env = DummyVectorEnv([lambda: ArbitrageEnv(market, amm, USD=args.USING_USD) for _ in range(args.training_num)])
    test_env = DummyVectorEnv([lambda: ArbitrageEnv(eval_market, amm, USD=args.USING_USD) for _ in range(args.test_num)])
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    args.USING_USD = False
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    print(f"max_action: {args.max_action}")
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net_a, args.action_shape, max_action=args.max_action, device=args.device).to(
        args.device,
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy: DDPGPolicy = DDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    buffer = ReplayBuffer(
        args.buffer_size,
        # buffer_num=len(train_env),
        # # ignore_obs_next=True,
        # # save_only_last_obs=True,
        # alpha=args.alpha,
        # beta=args.beta,
    )
    train_collector = Collector(policy, train_env, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_env, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ddpg"
    log_name = os.path.join(args.task, args.algo_name, str(args.fee_rate), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
        
    def save_dist_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "best_dist_policy.pth"))

    if not args.watch:
        # trainer
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            save_dist_fn=save_dist_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
            verbose=True,
        ).run()
        pprint.pprint(result)

#     # Let's watch its performance!
    test_env.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(collector_stats)


if __name__ == "__main__":
    
    fee_rates = [0.02, 0.1, 0.5, 0.04, 0.06, 0.08, 0.2]
    
    for fee_rate in fee_rates:
        for _ in range(5):
            test_ddpg(fee_rate)
        