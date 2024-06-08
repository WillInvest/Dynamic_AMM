import wandb
import yaml
import os

# Load the sweep configuration from the YAML file
with open('sweep.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="AMM-SB3-TD3")

# Define the training function
def train():
    # Run the main training script here
    os.system("python train.py")

# Run the sweep agent
wandb.agent(sweep_id, function=train, count=1)  # Adjust count as needed
