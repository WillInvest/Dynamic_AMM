import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

def alpha(gamma, S_t, sigma, delta_t, L):
    """Calculate alpha based on gamma."""
    return L * np.sqrt((1 - gamma) * S_t * np.exp(-0.25 * sigma**2 * delta_t))

def d1(gamma, P_t, S_t, sigma, delta_t):
    """Calculate d1 based on gamma."""
    return np.log((1 - gamma) * P_t / S_t) / (sigma * np.sqrt(delta_t))

def d2(gamma, P_t, S_t, sigma, delta_t):
    """Calculate d2 based on gamma."""
    return np.log(P_t / ((1 - gamma) * S_t)) / (sigma * np.sqrt(delta_t))

def d1_minus(gamma, P_t, S_t, sigma, delta_t):
    """Calculate d1- based on gamma."""
    return d1(gamma, P_t, S_t, sigma, delta_t) - 0.5 * sigma * np.sqrt(delta_t)

def d1_plus(gamma, P_t, S_t, sigma, delta_t):
    """Calculate d1+ based on gamma."""
    return d1(gamma, P_t, S_t, sigma, delta_t) + 0.5 * sigma * np.sqrt(delta_t)

def d2_plus(gamma, P_t, S_t, sigma, delta_t):
    """Calculate d2+ based on gamma."""
    return d2(gamma, P_t, S_t, sigma, delta_t) + 0.5 * sigma * np.sqrt(delta_t)

def d2_minus(gamma, P_t, S_t, sigma, delta_t):
    """Calculate d2- based on gamma."""
    return d2(gamma, P_t, S_t, sigma, delta_t) - 0.5 * sigma * np.sqrt(delta_t)

def calculate_Fin(gamma, params):
    """Calculate F(gamma) - the term inside parentheses in Fee_in."""
    X_t = params['X_t']
    Y_t = params['Y_t']
    P_t = Y_t / X_t
    theta = params['theta']
    S_t = P_t * (1-gamma) ** theta
    sigma = params['sigma']
    delta_t = params['delta_t']
    L = params['L']
    
    a = alpha(gamma, S_t, sigma, delta_t, L)
    term1 = a * (norm.cdf(d1(gamma, P_t, S_t, sigma, delta_t)) + 
                 norm.cdf(-d2(gamma, P_t, S_t, sigma, delta_t)))
    term2 = S_t * X_t * norm.cdf(d1_minus(gamma, P_t, S_t, sigma, delta_t))
    term3 = Y_t * norm.cdf(-d2_plus(gamma, P_t, S_t, sigma, delta_t))
    return term1 - term2 - term3

def calculate_Fout(gamma, params):
    """Calculate F(gamma) - the term inside parentheses in Fee_out."""
    X_t = params['X_t']
    Y_t = params['Y_t']
    P_t = Y_t / X_t
    theta = params['theta']
    S_t = P_t * (1-gamma) ** theta
    sigma = params['sigma']
    delta_t = params['delta_t']
    L = params['L']
    
    a = alpha(gamma, S_t, sigma, delta_t, L)
    term1 = a/(1-gamma) * (norm.cdf(d1(gamma, P_t, S_t, sigma, delta_t)) + 
                          norm.cdf(-d2(gamma, P_t, S_t, sigma, delta_t)))
    term2 = S_t * X_t * norm.cdf(-d2_minus(gamma, P_t, S_t, sigma, delta_t))
    term3 = Y_t * norm.cdf(d1_plus(gamma, P_t, S_t, sigma, delta_t))
    return -term1 + term2 + term3

def calculate_fee_in(gamma, params):
    """Calculate Fee_in for a given gamma."""
    return gamma / (1 - gamma) * calculate_Fin(gamma, params)

def calculate_fee_out(gamma, params):
    """Calculate Fee_out for a given gamma."""
    return gamma * calculate_Fout(gamma, params)

def find_optimal_fee_in(params):
    """Find optimal gamma to maximize the incoming fee."""
    def objective(gamma):
        if gamma <= 0 or gamma >= 1:
            return np.inf
        return -calculate_fee_in(gamma, params)
    
    result = optimize.minimize_scalar(objective, bounds=(1e-6, 0.0005), method='bounded', options={'xatol': 1e-12})
    return result.x, -result.fun

def find_optimal_fee_out(params):
    """Find optimal gamma to maximize the outgoing fee."""
    def objective(gamma):
        if gamma <= 0 or gamma >= 1:
            return np.inf
        return -calculate_fee_out(gamma, params)
    result = optimize.minimize_scalar(objective, bounds=(1e-6, 0.0005), method='bounded', options={'xatol': 1e-12})
    return result.x, -result.fun

def find_matching_fee_out(params, target_revenue, opt_gamma_in):
    """Find the outgoing fee rate that achieves the target revenue."""
    def objective(gamma):
        if gamma <= 0 or gamma >= 1:
            return np.inf
        fee_revenue = calculate_fee_out(gamma, params)
        return (fee_revenue - target_revenue) ** 2
    
    result = optimize.minimize_scalar(objective, bounds=(1e-5, opt_gamma_in), method='bounded', options={'xatol': 1e-120})
    return result.x

def main():
    # Define parameter ranges to test
    sigma_values = np.round(np.arange(0.03, 1.001, 0.001), 3)
    gamma_values = np.round(np.arange(0.0001, 0.9001, 0.0001), 4)
    theta_values = [0]
    base_params = {
        'X_t': 1e6,
        'Y_t': 1e6, 
        'delta_t': 12/365/24/60/60,  # Time interval = 12 seconds
        'L': 1e6
        }
    
    results_df = []
    total_combinations = len(sigma_values) * len(theta_values)
    progress_bar = tqdm(total=total_combinations, desc="Progress")
    # Run optimization for each parameter combination
    for sigma in sigma_values:
        for theta in theta_values:
            # Update parameters
            params = base_params.copy()
            params['sigma'] = sigma
            params['theta'] = theta
            # Find optimal incoming fee rate and its revenue
            opt_gamma_in, opt_revenue_in = find_optimal_fee_in(params)
            opt_gamma_out, opt_revenue_out = find_optimal_fee_out(params)
            
            # Find matching outgoing fee rate
            matching_gamma_out = find_matching_fee_out(params, opt_revenue_in, opt_gamma_in)
            
            # Add results to dataframe
            results_df.append({
                'sigma': sigma,
                'theta': theta,
                'opt_gamma_in': opt_gamma_in,
                'opt_revenue': opt_revenue_in,
                'opt_gamma_out': opt_gamma_out,
                'opt_revenue_out': opt_revenue_out,
                'matching_gamma_out': matching_gamma_out
            })
            progress_bar.update(1)
    progress_bar.close()
    # Convert to DataFrame and save results
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('fee_comparison_results.csv', index=False)
    print("Results saved to 'fee_comparison_results.csv'")
    
if __name__ == "__main__":
    main()