from parametric import ParametricValueModel
from tqdm import tqdm

def run_single_combination(params):
    """
    Run a single parameter combination
    
    Parameters:
    -----------
    params : tuple
        Tuple containing (L, k, mu, sigma, gamma, delta_t, num_sample, num_integration_points)
        
    Returns:
    --------
    result : dict
        Dictionary with parameters and results
    """
    import time
    start_time = time.time()
    L, k, mu, sigma, gamma, delta_t, num_sample, num_integration_points = params
    
    try:
        model = ParametricValueModel(L=L, mu=mu, sigma=sigma, gamma=gamma, delta_t=delta_t)
        optimal_cin, optimal_cout, _, _ = model.find_optimal_c_in_out(
            k=k,
            num_samples=num_sample, 
            num_integration_points=num_integration_points, 
            initial_c=1.0, 
            maximum=False
        )
        
        execution_time = time.time() - start_time
        return {
            'L': L,
            'k': k,
            'mu': mu,
            'sigma': sigma,
            'gamma': gamma,
            'delta_t': delta_t,
            'num_sample': num_sample,
            'num_integration_points': num_integration_points,
            'optimal_cin': optimal_cin,
            'optimal_cout': optimal_cout,
            'execution_time': execution_time,
            'status': 'success'
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            'L': L,
            'k': k,
            'mu': mu,
            'sigma': sigma,
            'gamma': gamma,
            'delta_t': delta_t,
            'num_sample': num_sample,
            'num_integration_points': num_integration_points,
            'error': str(e),
            'execution_time': execution_time,
            'status': 'error'
        }

def parallel_run(n_jobs=4, batch_size=500, results_dir=None):
    """
    Run parameter combinations in parallel with tqdm monitoring
    
    Parameters:
    -----------
    n_jobs : int
        Number of parallel jobs to run
    """
    import pandas as pd
    from tqdm.auto import tqdm
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    import time
    from tqdm.contrib.concurrent import process_map
    
    # Define parameter grid
    L_list = [100]
    k_list = [0]
    mu_list = [0.0, 0.2, 0.4]
    sigma_list = np.arange(0.01, 0.51, 0.01)
    gamma_list = np.arange(0.0005, 0.0505, 0.0005)
    delta_t_list = [1]
    num_sample_list = [100]
    num_integration_points_list = [100]
    
    # Create all parameter combinations
    param_combinations = []
    for L in L_list:
        for k in k_list:
            for mu in mu_list:
                for sigma in sigma_list:
                    for gamma in gamma_list:
                        for delta_t in delta_t_list:
                            for num_sample in num_sample_list:
                                for num_integration_points in num_integration_points_list:
                                    param_combinations.append(
                                        (L, k, mu, sigma, gamma, delta_t, num_sample, num_integration_points)
                                    )
    
    # Calculate total number of combinations
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations: {total_combinations}")
    
    # Initialize results list and checkpoint file
    results = []
    checkpoint_file = os.path.join(results_dir, 'checkpoint.csv')
    completed_params = set()
    
    # Check if checkpoint file exists and load already completed combinations
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        
        for _, row in checkpoint_df.iterrows():
            # Create a parameter signature to identify completed combinations
            param_sig = (
                row['L'], row['k'], row['mu'], row['sigma'], 
                row['gamma'], row['delta_t'], row['num_sample'], 
                row['num_integration_points']
            )
            completed_params.add(param_sig)
            results.append(row.to_dict())
        
        print(f"Loaded {len(completed_params)} completed parameter combinations from checkpoint")
    
    # Filter out already completed parameter combinations
    param_combinations = [p for p in param_combinations if p not in completed_params]
    print(f"Remaining parameter combinations to process: {len(param_combinations)}")
    
    # Function to save checkpoint
    def save_checkpoint():
        pd.DataFrame(results).to_csv(f'{results_dir}/checkpoint.csv', index=False)
        pd.DataFrame(results).to_csv(f'{results_dir}/results.csv', index=False)  # Final results file
    
    # Using tqdm.contrib.concurrent.process_map for built-in progress monitoring
    print("Starting parallel processing with tqdm monitoring...")
    
    # Create batches for better checkpointing
    print(f"Batch size: {batch_size} | length of param_combinations: {len(param_combinations)}")
    batches = [param_combinations[i:i + batch_size] for i in range(0, len(param_combinations), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        print(f"\nProcessing batch {batch_idx+1}/{len(batches)} ({len(batch)} combinations)")
        
        # Process batch in parallel with tqdm monitoring
        batch_results = process_map(
            run_single_combination, 
            batch,
            max_workers=n_jobs,
            desc=f"Batch {batch_idx+1}/{len(batches)}",
            chunksize=1,  # Process one task at a time for better progress reporting
            position=0,
            leave=True
        )
        
        # Add batch results to overall results
        results.extend(batch_results)
        
        # Save checkpoint after each batch
        save_checkpoint()
        print(f"Checkpoint saved after batch {batch_idx+1}/{len(batches)}")
        
        # Display some metrics about this batch
        success_count = sum(1 for r in batch_results if r.get('status') == 'success')
        error_count = sum(1 for r in batch_results if r.get('status') == 'error')
        print(f"Batch results: {success_count} successful, {error_count} errors")
        
        # Show some sample results from this batch if available
        if success_count > 0:
            sample = next((r for r in batch_results if r.get('status') == 'success'), None)
            if sample:
                print(f"Sample result - L: {sample['L']}, k: {sample['k']}, optimal_cin: {sample['optimal_cin']:.6f}, optimal_cout: {sample['optimal_cout']:.6f}")
    
    # Save final results
    save_checkpoint()
    print(f"Completed all {total_combinations} parameter combinations")
    print(f"Results saved to results.csv")

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    root_dir = '/home/shiftpub/Dynamic_AMM/inf_step_exp/parametric_approach/results_plots_v2'
    results_dir = os.path.join(root_dir, 'results')
    single_effect_plot_dir = os.path.join(root_dir, 'plots/single_effect')
    interaction_plot_dir = os.path.join(root_dir, 'plots/interaction')
    os.makedirs(single_effect_plot_dir, exist_ok=True)
    os.makedirs(interaction_plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Run parameter sweep in parallel with tqdm monitoring')
    parser.add_argument('--jobs', type=int, default=30, help='Number of parallel jobs')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for processing')
    args = parser.parse_args()
    
    print(f"Starting parallel run with {args.jobs} jobs")
    # parallel_run(n_jobs=args.jobs, batch_size=args.batch_size, results_dir=results_dir)
    
    
    # Set the style for plots
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['figure.figsize'] = [12, 8]

    def analyze_optimal_constants(results_df):
        """
        Simple analysis of optimal cin and cout values focusing only on 
        parameter effects and interactions.
    
        Parameters:
        -----------
        results_df : pandas DataFrame
            DataFrame containing results from parameter sweep with columns:
            L, k, mu, sigma, gamma, num_sample, num_integration_points,
            optimal_cin, optimal_cout
        """
        single_effect_plot_dir = f'{root_dir}/plots/single_effect'
        interaction_plot_dir = f'{root_dir}/plots/interaction'
        os.makedirs(single_effect_plot_dir, exist_ok=True)
        os.makedirs(interaction_plot_dir, exist_ok=True)
        
        results_df = results_df.copy()
    
        # Add calculated columns for ratio and difference
        results_df['ratio'] = results_df['optimal_cout'] / results_df['optimal_cin']
        results_df['difference'] = results_df['optimal_cout'] - results_df['optimal_cin']
    
        # Verify columns exist
        print("Available columns:", results_df.columns.tolist())
    
        # Summary statistics
        summary_stats = pd.DataFrame({
            'optimal_cin': results_df['optimal_cin'].describe(),
            'optimal_cout': results_df['optimal_cout'].describe(),
            'ratio': results_df['ratio'].describe(),
            'difference': results_df['difference'].describe()
        })
        print(summary_stats)
    
        # Relevant parameters
        parameters = ['L', 'k', 'mu', 'sigma', 'gamma', 'num_sample', 'num_integration_points']
    
        # 2.1 Single Parameter Analysis
        for param in parameters:
            # Create 2x2 subplot for each parameter
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Effect of {param} on optimal constants', fontsize=16)
        
            # Box plots
            sns.boxplot(x=param, y='optimal_cin', data=results_df, ax=axs[0, 0])
            axs[0, 0].set_title(f'Effect on optimal_cin')
        
            sns.boxplot(x=param, y='optimal_cout', data=results_df, ax=axs[0, 1])
            axs[0, 1].set_title(f'Effect on optimal_cout')
        
            sns.boxplot(x=param, y='ratio', data=results_df, ax=axs[1, 0])
            axs[1, 0].set_title(f'Effect on cin/cout ratio')
        
            sns.boxplot(x=param, y='difference', data=results_df, ax=axs[1, 1])
            axs[1, 1].set_title(f'Effect on cin-cout difference')
        
            # Properly rotate x-tick labels for all subplots
            for ax in axs.flatten():
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
            plt.tight_layout()
            plt.savefig(f'{single_effect_plot_dir}/{param}_effect.png', dpi=300)
            plt.close()
        
            # Calculate average effect
            param_effect = results_df.groupby(param)[['optimal_cin', 'optimal_cout', 
                                                     'ratio', 'difference']].mean().reset_index()
            print(f"\nAverage effect of {param}:")
            print(param_effect)
    
        print("\n==== 3. TWO-PARAMETER INTERACTION ANALYSIS ====\n")
    
        # Focus on the main parameters
        main_params = ['mu', 'sigma', 'gamma']
    
        # Create all possible pairs of parameters
        param_pairs = []
        for i, param1 in enumerate(main_params):
            for param2 in main_params[i+1:]:
                param_pairs.append((param1, param2))
    
        print(f"Analyzing {len(param_pairs)} parameter pairs")
    
        # Analyze interactions for all main parameter pairs
        for param1, param2 in param_pairs:
            print(f"\nAnalyzing interaction between {param1} and {param2}")
        
            # Create average values for each parameter combination
            interaction_data = results_df.groupby([param1, param2])[
                ['optimal_cin', 'optimal_cout', 'ratio', 'difference']
            ].mean().reset_index()
        
            # Create heatmaps
            fig, axs = plt.subplots(2, 2, figsize=(18, 16))
            fig.suptitle(f'Interaction between {param1} and {param2}', fontsize=18)
        
            metrics = ['optimal_cin', 'optimal_cout', 'ratio', 'difference']
        
            for i, metric in enumerate(metrics):
                row, col = i // 2, i % 2
            
                # Create pivot table for heatmap
                try:
                    pivot_data = pd.pivot_table(
                        interaction_data, 
                        values=metric,
                        index=param1, 
                        columns=param2,
                        aggfunc='mean'
                    )
                
                    # Choose appropriate colormap based on the metric
                    if metric == 'difference':
                        cmap = 'coolwarm'  # Diverging colormap for difference
                        center = 0
                    else:
                        cmap = 'viridis'  # Sequential colormap for others
                        center = None
                
                    sns.heatmap(
                        pivot_data, 
                        annot=False, 
                        cmap=cmap, 
                        center=center,
                        ax=axs[row, col], 
                        fmt='.2f',
                        cbar_kws={'label': metric}
                    )
                    axs[row, col].set_title(f'Effect on {metric}')
                
                    # Rotate x-axis labels if they're too crowded
                    if len(np.unique(interaction_data[param2])) > 5:
                        plt.setp(axs[row, col].get_xticklabels(), rotation=45)
                
                except Exception as e:
                    print(f"Error creating heatmap for {param1} vs {param2} ({metric}): {e}")
                    axs[row, col].text(0.5, 0.5, f"Error: {e}", 
                                      horizontalalignment='center',
                                      verticalalignment='center')
        
            plt.tight_layout()
            plt.savefig(f'{interaction_plot_dir}/{param1}_vs_{param2}_interaction.png', dpi=300)
            plt.close()
    
        print("\nAnalysis complete. All visualizations saved to PNG files.")

    # Load your results from the CSV file
    results_df = pd.read_csv(f'{results_dir}/results.csv')
    # Run the analysis
    analyze_optimal_constants(results_df)