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
        cumulative_fee = self.locals['infos'][0]['cumulative_fee']
        cumulative_pnl = self.locals['infos'][0]['cumulative_pnl']
        # total_pnl = self.locals['infos'][0]['total_pnl']
        # total_fee = self.locals['infos'][0]['total_fee']
        # actions = self.locals['actions'][0]
        
        actions = self.locals['infos'][0]['fee_rate']
        done = self.locals['dones'][0]
        # swap_rates = self.locals['infos'][0]['swap_rates']
        # urgent_levels = self.locals['infos'][0]['urgent_levels']
        reward = self.locals['rewards'][0]
        noise = self.model.action_noise  # Access the action noise object

        # # Prepare log data
        log_data = {
            "actions": actions,
            'reward': reward
        }
        
        # Add urgent_levels with a prefix
        # for mc, level in urgent_levels.items():
        #     wandb.log({f"urgent_level/trader_{mc}": level})        
            
        # # Log noise for each action
        # if isinstance(noise, NormalActionNoise):
        #     noise_values = noise.sigma
        #     wandb.log({
        #         "Noise/mean": np.mean(noise_values),
        #         "Noise/std": np.std(noise_values),
        #         "Noise/individual_values": noise_values.tolist()
        #     })
                
        if done:
            wandb.log({
                "cumulative_fee": cumulative_fee,
                "cumulative_pnl": cumulative_pnl
            })    
            
            # for trader_id in total_pnl.keys():
            #     wandb.log({
            #         f"total_pnl/trader_{trader_id}": total_pnl[trader_id],
            #         f"total_fee/trader_{trader_id}": total_fee[trader_id]
            #     })
        
        # Log the data to wandb
        wandb.log(log_data)
        # for trader_id in swap_rates.keys():
        #     wandb.log({f"swap_rate/trader_{trader_id}": swap_rates[trader_id]})

        return True

