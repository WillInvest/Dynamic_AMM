import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm
from int2 import TwoStepIntegrate

def find_optimal_fee_rate(sigma: float, initial_gamma: float = 0.0001) -> dict:
    """
    Find the optimal fee rate (gamma) that maximizes the total fee revenue over two steps.
    
    Parameters:
    -----------
    sigma : float
        Volatility parameter
    initial_gamma : float
        Initial guess for the fee rate optimization
        
    Returns:
    --------
    dict
        Dictionary containing optimization results:
        - opt_gamma: optimal fee rate
        - total_fee_revenue: total fee revenue at optimal gamma
        - first_step_fee: fee revenue from first step
        - second_step_fee: fee revenue from second step
        - first_step_pool_value: pool value after first step
        - second_step_pool_value: pool value after second step
    """
    
    def single_step_objective_function(gamma):
        """
        Objective function to minimize (negative of total fee revenue)
        """
        if gamma <= 0 or gamma >= 1:
            return np.inf  # Invalid range
        
        # Create TwoStepIntegrate instance with current gamma
        integrator = TwoStepIntegrate(gamma=gamma, sigma=sigma)
        
        # Calculate total fee revenue (negative because we're minimizing)
        first_step_fee = integrator.calculate_fee_revenue_single_step()
        
        return -first_step_fee  # Negative because we want to maximize
            
    
    def two_step_objective_function(gamma):
        """
        Objective function to minimize (negative of total fee revenue)
        """
        if gamma <= 0 or gamma >= 1:
            return np.inf  # Invalid range
            
        # Create TwoStepIntegrate instance with current gamma
        integrator = TwoStepIntegrate(gamma=gamma, sigma=sigma)
        
        # Calculate total fee revenue (negative because we're minimizing)
        first_step_fee = integrator.calculate_fee_revenue_single_step()
        second_step_fee = integrator.calculate_fee_revenue_second_step()
        total_fee = first_step_fee + second_step_fee
        
        return -total_fee  # Negative because we want to maximize
    
    # Run optimization
    single_step_result = optimize.minimize_scalar(
        single_step_objective_function,
        bounds=(1e-5, 0.0005),  # Same bounds as single step case
        method='bounded',
        options={'xatol': 1e-10}
    )
    
    two_step_result = optimize.minimize_scalar(
        two_step_objective_function,
        bounds=(1e-5, 0.0005),  # Same bounds as single step case
        method='bounded',
        options={'xatol': 1e-10}
    )

    # Get optimal gamma
    single_step_opt_gamma = single_step_result.x
    two_step_opt_gamma = two_step_result.x

    
    # Calculate all metrics at optimal gamma
    single_integrator = TwoStepIntegrate(gamma=single_step_opt_gamma, sigma=sigma)
    single_step_fee = single_integrator.calculate_fee_revenue_single_step()
    
    two_integrator = TwoStepIntegrate(gamma=two_step_opt_gamma, sigma=sigma)
    first_step_fee = two_integrator.calculate_fee_revenue_single_step()
    second_step_fee = two_integrator.calculate_fee_revenue_second_step()
    total_fee = first_step_fee + second_step_fee
    
    return {
        'sigma': sigma,
        'single_step_opt_gamma': single_step_opt_gamma,
        'single_step_fee': single_step_fee,
        'two_step_opt_gamma': two_step_opt_gamma,
        'first_step_fee': first_step_fee,
        'second_step_fee': second_step_fee,
        'two_step_fee_revenue': total_fee
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    # Create results directory if it doesn't exist
    results_path = '/home/shiftpub/Dynamic_AMM/CPMM/TwoStep/results'
    os.makedirs(results_path, exist_ok=True)
    
    # Define sigma values to test
    sigma_values = np.round(np.arange(0.1, 1.1, 0.01), 2)
    
    # Run optimization for each sigma
    results = []
    for sigma in tqdm(sigma_values, desc="Processing sigma values"):
        result = find_optimal_fee_rate(sigma)
        results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(results_path, 'two_step_optimal_fee_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to '{csv_path}'")
    
   