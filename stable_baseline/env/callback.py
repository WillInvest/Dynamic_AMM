import os
import csv
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, log_name='trade_log.csv', verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.log_file = log_name
        self.log_headers_written = False
        
        # Check if the log file exists and delete it
        if os.path.exists(self.log_file):
            os.remove(self.log_file)


    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]

        # Flattening arrays for CSV
        # details1_flat = self.flatten_dict(infos['details1'])
        # details2_flat = self.flatten_dict(infos['details2'])
        # details3_flat = self.flatten_dict(infos['details3'])


        # Log data including flattened details
        log_data = {
            "step": self.num_timesteps,
            "rew1": infos['rew1'],
            "rew2": infos['rew2'],
            "fee1": infos['fee1'],
            "fee2": infos['fee2'],
            "swap_rate1": infos['swap_rate1'],
            "swap_rate2": infos['swap_rate2'],
            "urgent_level": infos['urgent_level'],
            "total_rewards": infos['total_rewards'],
            "cumulative_fee": infos['cumulative_fee'],
            "amm_fee": self.training_env.get_attr('amm')[0].fee,
            "market_sigma": self.training_env.get_attr('market')[0].sigma,
        }

        # Add flattened details
        # log_data.update(details1_flat)
        # log_data.update(details2_flat)
        # log_data.update(details3_flat)

        # self.log_to_csv(log_data)

        wandb.log({
            "step": self.num_timesteps,
            "Rewards/rew1": infos['rew1'],
            "Rewards/rew2": infos['rew2'],
            "Fees/fee1": infos['fee1'],
            "Fees/fee2": infos['fee2'],
            "Swap Rates/swap_rate1": infos['swap_rate1'],
            "Swap Rates/swap_rate2": infos['swap_rate2'],
            # "Detail1/Details1": infos['details1'],
            # "Detail2/Details2": infos['details2'],
            # "Detail3/Details3": infos['details3'],
            "Misc/urgent_level": infos['urgent_level'],
            "Misc/amm_fee": self.training_env.get_attr('amm')[0].fee,
            "Misc/market_sigma": self.training_env.get_attr('market')[0].sigma
        })

        return True

    def flatten_dict(self, d, parent_key='', sep='_'):
        """
        Flatten a nested dictionary, using `sep` as the separator between keys.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, val in enumerate(v):
                    items.append((f"{new_key}{sep}{i}", val))
            else:
                items.append((new_key, v))
        return dict(items)

    def log_to_csv(self, log_data):
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
            if not file_exists or not self.log_headers_written:
                writer.writeheader()
                self.log_headers_written = True
            writer.writerow(log_data)
