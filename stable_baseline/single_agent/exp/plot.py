import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_fee_cross_model_fix_rate(csv_path, s=None):

    csv_files = [f for f in os.listdir(csv_path) if (f.endswith(f'{s}.csv'))]
    csv_files.sort(key=lambda x: float(x.split('_')[0].split('t')[1]))
    
    model_length = len(csv_files)
    print(f"there are {model_length} models")
    fig, axes = plt.subplots(model_length, 1, figsize=(20, 10))
    for i, file in enumerate(csv_files):
        file_path = os.path.join(csv_path, file)
        df = pd.read_csv(file_path)
        # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
        sorted_df = df.sort_values(by=['fee_rate', 'sigma', 'iterations'])
        # Calculate relative fee based on the condition
        sorted_df['relative_fee'] = np.where(sorted_df['reward'] < 0, 0, sorted_df['fee'])
        sigma_df = sorted_df[sorted_df['sigma'] == s]
        sns.lineplot(data=sigma_df, x='fee_rate', y='relative_fee', ax=axes[i])
        axes[i].set_title(f'model - {file}')
        axes[i].set_xlabel('Fee Rate')
        axes[i].set_ylabel('Relative Fee')
        # Find the fee rate with the highest relative fee
        mean_relative_fee = sigma_df.groupby('fee_rate')['relative_fee'].mean()
        # Find the fee rate with the highest average relative fee
        max_fee_rate = mean_relative_fee.idxmax()
        # Draw a vertical line at the fee rate with the highest relative fee
        axes[i].axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
        axes[i].legend()
        axes[i].grid(True)
        
    fig.suptitle(f'Relative Fee vs. Fee Rate for same Sigma {s:.1f} with different models', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(csv_path, f"fee_cross_sigma_fix_rate_with_sigma_{s}.png")
    plt.savefig(plot_path)


def plot_fee_cross_model_fix_sigma(csv_path, r=None, s=None):

    csv_files = [f for f in os.listdir(csv_path) if (f.startswith(f'agent{r:.2f}') and not f.endswith('1.0.csv'))]
    csv_files.sort(key=lambda x: float(x.split('_')[2].split('.csv')[0]))
    
    model_length = len(csv_files)
    print(f"there are {model_length} models")
    fig, axes = plt.subplots(model_length, 1, figsize=(20, 10))
    for i, file in enumerate(csv_files):
        file_path = os.path.join(csv_path, file)
        df = pd.read_csv(file_path)
        # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
        sorted_df = df.sort_values(by=['fee_rate', 'sigma', 'iterations'])
        # Calculate relative fee based on the condition
        sorted_df['relative_fee'] = np.where(sorted_df['reward'] < 0, 0, sorted_df['fee'])
        sigma_df = sorted_df[sorted_df['sigma'] == s]
        sns.lineplot(data=sigma_df, x='fee_rate', y='relative_fee', ax=axes[i])
        axes[i].set_title(f'model - {file}')
        axes[i].set_xlabel('Fee Rate')
        axes[i].set_ylabel('Relative Fee')
        axes[i].grid(True)
        # Find the fee rate with the highest relative fee
        mean_relative_fee = sigma_df.groupby('fee_rate')['relative_fee'].mean()
        # Find the fee rate with the highest average relative fee
        max_fee_rate = mean_relative_fee.idxmax()
        # Draw a vertical line at the fee rate with the highest relative fee
        axes[i].axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
        axes[i].legend()

    fig.suptitle(f'Relative Fee vs. Fee Rate for same Sigma {s:.1f} with different models', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(csv_path, f"fee_cross_sigma_fix_sigma_r_{r}_s_{s}.png")
    plt.savefig(plot_path)

def plot_fee_cross_sigma(csv_path, r, s):
    
    csv_files = [f for f in os.listdir(csv_path) if f.endswith(f'{s:.1f}.csv') and f.startswith(f'agent{r:.2f}')]
    print(f"csv_files: {csv_files}")
    
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    # Iterate over each .csv file and append its contents to the combined DataFrame
    for file in csv_files:
        file_path = os.path.join(csv_path, file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
    sorted_df = combined_df.sort_values(by=['fee_rate', 'sigma', 'iterations'])

    # Calculate relative fee based on the condition
    sorted_df['relative_fee'] = np.where(sorted_df['reward'] < 0, 0, sorted_df['fee'])

    # Get unique sigma values
    sigma_values = sorted_df['sigma'].unique()

    # Filter the DataFrame to only include the data for the first sigma value
    sigma_df = sorted_df[sorted_df['sigma'] == sigma_values[0]]

    fig, axes = plt.subplots(len(sigma_values), 1, figsize=(20, 10))

    for i, sigma in enumerate(sigma_values):
        sigma_df = sorted_df[sorted_df['sigma'] == sigma]
        sns.lineplot(data=sigma_df, x='fee_rate', y='relative_fee', ax=axes[i])
        axes[i].set_title(f'Sigma = {sigma}')
        axes[i].set_xlabel('Fee Rate')
        axes[i].set_ylabel('Relative Fee')
        axes[i].grid(True)
        # Find the fee rate with the highest relative fee
        mean_relative_fee = sigma_df.groupby('fee_rate')['relative_fee'].mean()
        # Find the fee rate with the highest average relative fee
        max_fee_rate = mean_relative_fee.idxmax()
        # Draw a vertical line at the fee rate with the highest relative fee
        axes[i].axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
        axes[i].legend()

    fig.suptitle(f'Relative Fee vs. Fee Rate for Different Sigma Values - with the same model - r:{r:.2f} - s: {s:.2f}', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(csv_path, f"fee_cross_sigma_r{r:.2f}_s{s:.2f}.png")
    plt.savefig(plot_path)
    

if __name__ == "__main__":
    # Get a list of all .csv files in the directory
    csv_path = "/home/shiftpub/AMM-Python/stable_baseline/single_agent/models/TD3/2024-06-10_10-08-52/csv_file_success"

    # Call the function to plot the fee data
    # plot_fee_cross_sigma(csv_path, r=0.01, s=0.2)
    
    plot_fee_cross_model_fix_sigma(csv_path, r=0.11, s=0.6)
    
    # plot_fee_cross_model_fix_rate(csv_path, s=0.2)
    
    
    
