import pandas as pd
import os
import glob

# Path to the directory containing the CSV files
input_dir = '/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/mc_results'
output_file = '/home/shiftpub/Dynamic_AMM/inf_step_exp/mc_approach/combined_results.csv'

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(input_dir, 'results_sigma*.csv'))

# Sort the files to ensure consistent order
csv_files.sort()

print(f"Found {len(csv_files)} CSV files to combine")

# Initialize an empty list to store dataframes
dfs = []

# Read each CSV file and append to the list
for file in csv_files:
    print(f"Reading {file}")
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate all dataframes
print("Concatenating all dataframes...")
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
print(f"Saving combined data to {output_file}")
combined_df.to_csv(output_file, index=False)

print(f"Combined {len(csv_files)} files into {output_file}")
print(f"Total rows in combined file: {len(combined_df)}") 