import matplotlib.pyplot as plt
import seaborn as sns


def plot_total_pnls(total_pnls_constant, total_pnls_rl):
    # Prepare data for plotting
    data = []
    labels = []
    for fee_rate, pnls in total_pnls_constant.items():
        data.append(pnls)
        labels.append(f'Fee Rate {fee_rate:.4f}')
    data.append(total_pnls_rl)
    labels.append('RL-based AMM')

    # Create the box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Total PnL')
    plt.title('Total PnL for Different Fee Structures')
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
    plt.savefig('total_pnl_boxplot.png')
    
def plot_total_fees(total_fees_constant, total_fees_rl):
    # Prepare data for plotting
    data = []
    labels = []
    for fee_rate, fees in total_fees_constant.items():
        data.append(fees)
        labels.append(f'Fee Rate {fee_rate:.4f}')
    data.append(total_fees_rl)
    labels.append('RL-based AMM')
    # Create the box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Total Fee')
    plt.title('Total Fee for Different Fee Structures')
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
    plt.savefig('total_fee_boxplot.png')
    
    
def plot_total_vols(total_vols_constant, total_vols_rl):
    # Prepare data for plotting
    data = []
    labels = []
    for fee_rate, fees in total_vols_constant.items():
        data.append(fees)
        labels.append(f'Fee Rate {fee_rate:.4f}')
    data.append(total_vols_rl)
    labels.append('RL-based AMM')
    # Create the box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Total Trade Counts')
    plt.title('Total trade counts for Different Fee Structures')
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
    plt.savefig('total_vol_boxplot.png')

def plot_total_dynamic_fee(total_dynamic_fee):
    plt.figure(figsize=(12, 8))
    # Determine the unique fee rates and number of bins
    unique_fee_rates = set(total_dynamic_fee)
    num_bins = len(unique_fee_rates)
    # Create a histogram with the number of bins equal to the unique fee rates
    sns.histplot(total_dynamic_fee, bins=num_bins, kde=False)
    # Set x and y labels
    plt.xlabel('Fee Rates')
    plt.ylabel('Frequency')
    # Set title
    plt.title('Distribution of Dynamic Fee Rates')
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    # Display grid
    plt.grid(True)
    # Tight layout for better spacing
    plt.tight_layout()
    # Save the plot
    plt.savefig('dynamic_fee_distribution.png')
    # Show the plot
    plt.show()
    
def plot_total_price_distance(total_price_distance_constant, total_price_distance_rl):
    # Prepare data for plotting
    data = []
    labels = []
    for fee_rate, pnls in total_price_distance_constant.items():
        data.append(pnls)
        labels.append(f'Fee Rate {fee_rate:.4f}')
    data.append(total_price_distance_rl)
    labels.append('RL-based AMM')

    # Create the box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Total price_distance')
    plt.title('Total price_distance for Different Fee Structures')
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
    plt.savefig('total_price_distance_boxplot.png')