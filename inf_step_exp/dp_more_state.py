import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import ParametricValueModel
from time import time

def optimized_single_thread_sweep():
    start_time = time()
    
    # Define parameter ranges - adjust the number of points as needed
    mu_values = np.linspace(0, 2, 40)
    sigma_values = np.linspace(0.01, 2, 40)
    gamma_values = np.linspace(0, 0.90, 40)
    
    # Pre-compute total size
    total_combinations = len(mu_values) * len(sigma_values) * len(gamma_values)
    print(f"Processing {total_combinations} parameter combinations...")
    
    # Pre-allocate results arrays - adding one more column for expected difference
    results = np.zeros((total_combinations, 6))
    
    # Process all combinations in a single loop with vectorized inner operations
    idx = 0
    for mu in mu_values:
        for sigma in sigma_values:
            for gamma in gamma_values:
                # Initialize model
                model = ParametricValueModel(mu, sigma, gamma)
                
                # Generate data efficiently
                df = model.generate_raw_data(L=1000000, num_samples=1000)
                
                # Vectorized calculations on NumPy arrays for better performance
                original_pool = df['original_pool'].values
                discounted_outgoing = df['discounted_outgoing_value'].values
                discounted_incoming = df['discounted_incoming_value'].values
                
                # Direct array operations
                original_out = original_pool - discounted_outgoing
                original_in = original_pool - discounted_incoming
                in_out_diff = discounted_incoming - discounted_outgoing
                
                # Calculate expected difference (mean of the differences)
                expected_diff = np.mean(in_out_diff)
                
                # Count values with NumPy operations
                num_out_high = np.sum(original_out < 0)
                num_in_high = np.sum(original_in < 0)
                
                # Store results
                results[idx] = [mu, sigma, gamma, num_out_high, num_in_high, expected_diff]
                idx += 1
                
                # Optional progress tracking (print every 1000 iterations)
                if idx % 1000 == 0 or idx == total_combinations:
                    print(f"Processed {idx}/{total_combinations} combinations ({idx/total_combinations*100:.1f}%)")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(
        results, 
        columns=['mu', 'sigma', 'gamma', 'num_positive_out', 'num_positive_in', 'expected_diff']
    )
    
    # Save results
    results_df.to_csv('results.csv', index=False)
    
    # Print summary
    print(f"\nComputation completed in {time() - start_time:.2f} seconds")
    print(results_df.head().to_markdown())
    print(f"All results saved to results.csv")
    
    return results_df

# Alternative approach that uses batched processing to better control memory usage
def batched_vectorized_sweep():
    start_time = time()
    
    # Define parameter ranges
    mu_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
    sigma_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
    gamma_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
    
    # Create all parameter combinations (more memory efficient than meshgrid)
    param_combinations = [(mu, sigma, gamma) 
                         for mu in mu_values 
                         for sigma in sigma_values 
                         for gamma in gamma_values]
    
    total_combinations = len(param_combinations)
    print(f"Processing {total_combinations} parameter combinations...")
    
    # Process in batches to control memory usage
    batch_size = 1000  # Adjust based on your available memory
    num_batches = (total_combinations // batch_size) + (1 if total_combinations % batch_size != 0 else 0)
    
    # Storage for all results
    all_results = []
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_combinations)
        
        batch_params = param_combinations[start_idx:end_idx]
        batch_results = []
        
        print(f"Processing batch {batch_num+1}/{num_batches} ({start_idx}-{end_idx})...")
        
        for mu, sigma, gamma in batch_params:
            # Initialize model
            model = ParametricValueModel(mu, sigma, gamma)
            
            # Generate data efficiently
            df = model.generate_raw_data(L=1000000, num_samples=1000)
            
            # Get numpy arrays for better performance
            incoming = df['discounted_incoming_value'].values
            outgoing = df['discounted_outgoing_value'].values
            original = df['original_pool'].values
            
            # Vectorized comparison operations
            num_out_high = np.sum((original - outgoing) < 0)
            num_in_high = np.sum((original - incoming) < 0)
            
            # Calculate expected difference between incoming and outgoing
            expected_diff = np.mean(incoming - outgoing)
            
            # Store results for this parameter set
            batch_results.append((mu, sigma, gamma, num_out_high, num_in_high, expected_diff))
        
        # Add batch results to overall results
        all_results.extend(batch_results)
        
        print(f"Batch {batch_num+1} complete. {len(all_results)}/{total_combinations} combinations processed.")
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(
        all_results, 
        columns=['mu', 'sigma', 'gamma', 'num_positive_out', 'num_positive_in', 'expected_diff']
    )
    
    # Save results
    results_df.to_csv('results.csv', index=False)
    
    # Print summary
    print(f"\nComputation completed in {time() - start_time:.2f} seconds")
    print(results_df.head().to_markdown())
    print(f"All results saved to results.csv")
    
    return results_df

if __name__ == "__main__":
    # Choose one of these methods
    # optimized_single_thread_sweep()  # Simple single-threaded approach
    batched_vectorized_sweep()  # Better memory management