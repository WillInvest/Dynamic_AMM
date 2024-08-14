import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
data = pd.read_csv('/Users/haofu/AMM-Python/exp/final_results/total_price_distance.csv')
data = data.iloc[:, 1:]  # Remove the first column if necessary
labels = data.columns
print(f"Labels: {labels}")

# Each row will correspond to a single observation for a given fee rate
data_melt = pd.melt(data, var_name='Fee_Rate', value_name='Cumulative_Fee')

# Display the reshaped data for verification
print(data_melt.head(10))

# Ordinary Least Squares model: Total_PnL ~ C(Fee_Rate) means 'Total_PnL' is explained by the categorical variable 'Fee_Rate'
model = ols('Cumulative_Fee ~ C(Fee_Rate)', data=data_melt).fit()

# Get ANOVA table
anova_table = sm.stats.anova_lm(model, typ=2)

# Display the ANOVA table
print(anova_table)

# Format the ANOVA results into a table
anova_results = {
    "Source": ["C(Fee_Rate)", "Residual"],
    "Sum of Squares": [anova_table['sum_sq'][0], anova_table['sum_sq'][1]],
    "Degrees of Freedom": [anova_table['df'][0], anova_table['df'][1]],
    "F-statistic": [anova_table['F'][0], None],
    "p-value": [anova_table['PR(>F)'][0], None]
}

# Create a DataFrame for the ANOVA results
anova_results_df = pd.DataFrame(anova_results)

# Display the table for easy copying
print(anova_results_df.to_string(index=False))

# Set plot size and font sizes
plt.figure(figsize=(12, 8))  # Increase the figure size for better readability

# Adjusting font sizes
plt.rc('font', size=14)  # Default text size
plt.rc('axes', titlesize=16)  # Title size
plt.rc('axes', labelsize=14)  # Axes labels size
plt.rc('xtick', labelsize=12)  # X-tick labels size
plt.rc('ytick', labelsize=12)  # Y-tick labels size

# Create the box plot
plt.boxplot(data, labels=labels)
plt.ylabel('Total PnL')
plt.title('Total PnL for Different Fee Structures')

# Rotate x-axis labels for better readability
plt.xticks(rotation=20, ha='right')

# Add grid lines for better visual separation
plt.grid(True)

# Save the figure before showing
plt.savefig('/Users/haofu/AMM-Python/exp/final_results/total_price_distance.png', bbox_inches='tight')

# Display the plot
# plt.show()