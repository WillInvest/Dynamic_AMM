from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import wandb
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(self, log_name='trade_log.csv', verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.log_file = log_name
        self.log_headers_written = False


    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        done = self.locals['dones'][0]
        actions = self.locals['actions'][0]  # Get current actions
        noise = self.model.action_noise  # Access the action noise object
        
        # Log step details to WandB
        wandb.log({
            "step": self.num_timesteps,
            "urgent_level": infos['urgent_level'],
            "Rewards/rew1": infos['rew1'],
            "Rewards/rew2": infos['rew2'],
            "Fees/fee1": infos['fee1'],
            "Fees/fee2": infos['fee2'],
            "Swap Rates/swap_rate1": infos['swap_rate1'],
            "Swap Rates/swap_rate2": infos['swap_rate2'],
            "actions": actions,  # Log actions
        })

        # Log noise for each action
        if isinstance(noise, NormalActionNoise):
            noise_values = noise.sigma
            wandb.log({
                "Noise/mean": np.mean(noise_values),
                "Noise/std": np.std(noise_values),
                "Noise/individual_values": noise_values.tolist()
            })
        
        if done:
            wandb.log({
                "cumulative_fee": infos['cumulative_fee'],
                "cumulative_reward": infos['cumulative_reward']
            })

        return True
