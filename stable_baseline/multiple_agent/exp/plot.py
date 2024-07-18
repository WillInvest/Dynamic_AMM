import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

MIN_FEE_RATE = 0.0001  # 0.01%
MAX_FEE_RATE = 0.005   # 0.3%

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
RISK_AVERSION_VALUES = [0.2, 0.4, 0.6]


def get_csv_files(csv_path):
    # read and combine all csv files
    csv_names = [f for f in os.listdir(csv_path) if f.endswith('.csv') and any(f'risk_aversion_{rav}_' in f for rav in RISK_AVERSION_VALUES)]
    combined_df = pd.DataFrame()
    for csv_name in csv_names:
        csv_file = os.path.join(csv_path, csv_name)
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    # combined_df = combined_df[combined_df['reward'] >= -0.1].copy()
    combined_df = combined_df[combined_df['total_reward'] >= combined_df['total_reward'].quantile(0.25)].copy()

    filtered_df = combined_df
    # [combined_df['reward'] >= 0]
    
    # Apply min-max normalization to 'total_gas'
    min_gas = filtered_df['total_gas'].min()
    max_gas = filtered_df['total_gas'].max()
    filtered_df.loc[:, 'normalized_gas_fee'] = MIN_FEE_RATE + (MAX_FEE_RATE - MIN_FEE_RATE) * ((filtered_df['total_gas'] - min_gas) / (max_gas - min_gas))
    filtered_df = filtered_df[filtered_df['normalized_gas_fee'] >= filtered_df['fee_rate']]

    # filtered_df = filtered_df[filtered_df['model_fee_rate'] >= filtered_df['fee_rate']]

    sorted_df = filtered_df.sort_values(by=['model_fee_rate',
                                            'model_sigma',
                                            "fee_rate",
                                            "sigma",
                                            'iterations'])
    
    return sorted_df


def get_plot(csv_file, col_names, csv_path):
    for col_name in col_names:
        cols = sorted(csv_file[col_name].unique())
        num_cols = len(cols)
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_cols)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.flatten()  # Flatten the grid for easy iteration
        
        for i, col in enumerate(cols):
            df = csv_file[(csv_file[col_name] == col)]
            sns.boxplot(data=df, x='fee_rate', y='fee', ax=axes[i])
            axes[i].set_title(f'{col_name} = {col}')
            axes[i].set_xlabel('Fee Rate')
            axes[i].set_ylabel('Fee')
            axes[i].grid(True)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'Fee vs. Fee Rate for Different {col_name}', fontsize=16)
        plt.tight_layout()
        plot_path = os.path.join(csv_path, f"min_max_fee_cross_{col_name}.png")
        plt.savefig(plot_path)
        print(f"fee_cross_{col_name}.png saved")
        plt.close(fig)
        
    

if __name__ == "__main__":
    # Get a list of all .csv files in the directory
    csv_path = '/Users/haofu/AMM-Python/stable_baseline/multiple_agent/models/csv_file'
    col_names = ['model_sigma', 'sigma']
    csv_file = get_csv_files(csv_path)
    get_plot(csv_file, col_names, csv_path)

    
    
