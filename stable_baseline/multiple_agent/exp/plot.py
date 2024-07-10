import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = -0.1

def plot_model_fee_rate(csv_path):
    csv_names = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    # concate all csv files
    combined_df = pd.DataFrame()
    for csv_name in csv_names:
        csv_file = os.path.join(csv_path, csv_name)
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df = combined_df[(combined_df['model_sigma'] != 0.8) & (combined_df['sigma'] != 0.8)]
    # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
    sorted_df = combined_df.sort_values(by=['model_fee_rate',
                                            'model_sigma',
                                            "fee_rate",
                                            "sigma",
                                            'iterations'])
    sorted_df['relative_fee'] = np.where(sorted_df['total_reward'] < THRESHOLD, 0, sorted_df['fee'])
    
    # draw plot for model with different model_fee_rate
    model_fee_rates = sorted_df['model_fee_rate'].unique()
    fig, axes = plt.subplots(len(model_fee_rates), 1, figsize=(20, 10))
    for i, model_fee_rate in enumerate(model_fee_rates):
        model_df = sorted_df[sorted_df['model_fee_rate'] == model_fee_rate]
        sns.lineplot(data=model_df, x='fee_rate', y='relative_fee', ax=axes[i])
        axes[i].set_title(f'Model Fee Rate = {model_fee_rate}')
        axes[i].set_xlabel('Fee Rate')
        axes[i].set_ylabel('Relative Fee')
        axes[i].grid(True)
        # Find the fee rate with the highest relative fee
        mean_relative_fee = model_df.groupby('fee_rate')['relative_fee'].mean()
        # Find the fee rate with the highest average relative fee
        max_fee_rate = mean_relative_fee.idxmax()
        # Draw a vertical line at the fee rate with the highest relative fee
        axes[i].axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
        axes[i].legend()
    fig.suptitle(f'Relative Fee vs. Fee Rate for Different Model Fee Rates', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(csv_path, f"fee_cross_model_fee_rate.png")
    plt.savefig(plot_path)
    
def plot_model_sigma(csv_path):
    csv_names = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    # concate all csv files
    combined_df = pd.DataFrame()
    for csv_name in csv_names:
        csv_file = os.path.join(csv_path, csv_name)
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
    combined_df = combined_df[(combined_df['model_sigma'] != 0.8) & (combined_df['sigma'] != 0.8)]
    sorted_df = combined_df.sort_values(by=['model_fee_rate',
                                            'model_sigma',
                                            "fee_rate",
                                            "sigma",
                                            'iterations'])
    sorted_df['relative_fee'] = np.where(sorted_df['total_reward'] < THRESHOLD, 0, sorted_df['fee'])
    # draw plot for model with different model_fee_rate
    model_sigma = sorted_df['model_sigma'].unique()
    fig, axes = plt.subplots(len(model_sigma), 1, figsize=(20, 10))
    for i, model_sigma in enumerate(model_sigma):
        model_df = sorted_df[sorted_df['model_sigma'] == model_sigma]
        sns.lineplot(data=model_df, x='fee_rate', y='relative_fee', ax=axes[i])
        axes[i].set_title(f'Model Sigma = {model_sigma}')
        axes[i].set_xlabel('Fee Rate')
        axes[i].set_ylabel('Relative Fee')
        axes[i].grid(True)
        # Find the fee rate with the highest relative fee
        mean_relative_fee = model_df.groupby('fee_rate')['relative_fee'].mean()
        # Find the fee rate with the highest average relative fee
        max_fee_rate = mean_relative_fee.idxmax()
        # Draw a vertical line at the fee rate with the highest relative fee
        axes[i].axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
        axes[i].legend()
    fig.suptitle(f'Relative Fee vs. Fee Rate for Different Model Volatility', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(csv_path, f"fee_cross_model_volatility.png")
    plt.savefig(plot_path)
    
    
def plot_market_sigma(csv_path):
    csv_names = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    # concate all csv files
    combined_df = pd.DataFrame()
    for csv_name in csv_names:
        csv_file = os.path.join(csv_path, csv_name)
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df = combined_df[(combined_df['model_sigma'] != 0.8) & (combined_df['sigma'] != 0.8)]
    # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
    sorted_df = combined_df.sort_values(by=['model_fee_rate',
                                            'model_sigma',
                                            "fee_rate",
                                            "sigma",
                                            'iterations'])
    sorted_df['relative_fee'] = np.where(sorted_df['total_reward'] < THRESHOLD, 0, sorted_df['fee'])
    # draw plot for model with different model_fee_rate
    market_sigma = sorted_df['sigma'].unique()
    fig, axes = plt.subplots(len(market_sigma), 1, figsize=(20, 10))
    for i, market_sigma in enumerate(market_sigma):
        model_df = sorted_df[sorted_df['sigma'] == market_sigma]
        sns.lineplot(data=model_df, x='fee_rate', y='relative_fee', ax=axes[i])
        axes[i].set_title(f'Market Sigma = {market_sigma}')
        axes[i].set_xlabel('Fee Rate')
        axes[i].set_ylabel('Relative Fee')
        axes[i].grid(True)
        # Find the fee rate with the highest relative fee
        mean_relative_fee = model_df.groupby('fee_rate')['relative_fee'].mean()
        # Find the fee rate with the highest average relative fee
        max_fee_rate = mean_relative_fee.idxmax()
        # Draw a vertical line at the fee rate with the highest relative fee
        axes[i].axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
        axes[i].legend()
    fig.suptitle(f'Relative Fee vs. Fee Rate for Different Market Volatility', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(csv_path, f"fee_cross_market_volatility.png")
    plt.savefig(plot_path)
    
def plot_aggregate(csv_path):
    csv_names = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    # Concatenate all CSV files
    combined_df = pd.DataFrame()
    for csv_name in csv_names:
        csv_file = os.path.join(csv_path, csv_name)
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df = combined_df[(combined_df['model_sigma'] != 0.8) & (combined_df['sigma'] != 0.8)]
    # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
    sorted_df = combined_df.sort_values(by=['model_fee_rate',
                                            'model_sigma',
                                            "fee_rate",
                                            "sigma",
                                            'iterations'])
    sorted_df['relative_fee'] = np.where(sorted_df['total_reward'] < THRESHOLD, 0, sorted_df['fee'])
    
    # Calculate the fee rate with the highest average relative fee
    mean_relative_fee = sorted_df.groupby('fee_rate')['relative_fee'].mean()
    max_fee_rate = mean_relative_fee.idxmax()

    # Draw aggregate plot
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=sorted_df, x='fee_rate', y='relative_fee')
    plt.axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
    plt.title('Relative Fee vs. Fee Rate')
    plt.xlabel('Fee Rate')
    plt.ylabel('Relative Fee')
    plt.grid(True)
    plt.legend(title='Sigma and Model Fee Rate')
    
    plot_path = os.path.join(csv_path, f"aggregate_fee_rate_plot.png")
    plt.savefig(plot_path)
    plt.show()

    
def plot_total_gas(csv_path):
    csv_names = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    # Concatenate all CSV files
    combined_df = pd.DataFrame()
    for csv_name in csv_names:
        csv_file = os.path.join(csv_path, csv_name)
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_df = combined_df[(combined_df['model_sigma'] != 0.8) & (combined_df['sigma'] != 0.8)]
    # Sort the combined DataFrame by 'fee_rate', 'sigma', and 'iterations'
    sorted_df = combined_df.sort_values(by=['model_fee_rate',
                                            'model_sigma',
                                            "fee_rate",
                                            "sigma",
                                            'iterations'])
    sorted_df['relative_gas'] = np.where(sorted_df['total_reward'] < THRESHOLD, 0, sorted_df['total_gas'])
    
    # Calculate the fee rate with the highest average relative gas
    mean_relative_gas = sorted_df.groupby('fee_rate')['relative_gas'].mean()
    # max_fee_rate = mean_relative_gas.idxmax()

    # Draw aggregate plot
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=sorted_df, x='fee_rate', y='relative_gas')
    # plt.yscale('log')
    # plt.axvline(x=max_fee_rate, color='red', linestyle='--', label=f'Max Fee Rate: {max_fee_rate:.2f}')
    plt.title('Relative Gas vs. Fee Rate')
    plt.xlabel('Fee Rate')
    plt.ylabel('Relative Gas')
    plt.grid(True)
    # plt.legend(title='Sigma and Model Fee Rate')
    
    plot_path = os.path.join(csv_path, f"aggregate_gas_rate_plot.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    # Get a list of all .csv files in the directory
    csv_path = "/home/shiftpub/AMM-Python/stable_baseline/multiple_agent/models/csv_file"

    # Call the function to plot the fee data
    # plot_fee_cross_sigma(csv_path, r=0.01, s=0.2)
    
    # plot_model_fee_rate(csv_path)
    # plot_model_sigma(csv_path)
    # plot_market_sigma(csv_path)
    plot_total_gas(csv_path)

    # plot_fee_cross_model_fix_rate(csv_path, s=0.2)
    
    
    
