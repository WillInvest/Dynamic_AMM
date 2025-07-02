import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

time_scale = 1/365/24/60/60 # 1 second

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
    S_t = params['S_t']
    X_t = params['X_t']
    Y_t = params['Y_t']
    annual_sigma = params['sigma']
    sigma = annual_sigma * np.sqrt(time_scale)
    delta_t = params['delta_t']
    L = params['L']
    P_t = params['P_t']
    
    a = alpha(gamma, S_t, sigma, delta_t, L)
    term1 = a * (norm.cdf(d1(gamma, P_t, S_t, sigma, delta_t)) + 
                 norm.cdf(-d2(gamma, P_t, S_t, sigma, delta_t)))
    term2 = S_t * X_t * norm.cdf(d1_minus(gamma, P_t, S_t, sigma, delta_t))
    term3 = Y_t * norm.cdf(-d2_plus(gamma, P_t, S_t, sigma, delta_t))
    return term1 - term2 - term3

def calculate_Fout(gamma, params):
    """Calculate F(gamma) - the term inside parentheses in Fee_out."""
    S_t = params['S_t']
    X_t = params['X_t']
    Y_t = params['Y_t']
    annual_sigma = params['sigma']
    sigma = annual_sigma * np.sqrt(time_scale)
    delta_t = params['delta_t']
    L = params['L']
    P_t = params['P_t']
    
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
    
    result = optimize.minimize_scalar(objective, bounds=(1e-10, 0.9999), method='bounded')
    return result.x, -result.fun

def find_matching_fee_out(params, target_revenue, opt_gamma_in):
    """Find the outgoing fee rate that achieves the target revenue."""
    def objective(gamma):
        if gamma <= 0 or gamma >= 1:
            return np.inf
        fee_revenue = calculate_fee_out(gamma, params)
        return (fee_revenue - target_revenue) ** 2
    
    result = optimize.minimize_scalar(objective, bounds=(1e-10, opt_gamma_in), method='bounded')
    return result.x

def main():
    # Define parameter ranges to test
    sigma_values = np.round(np.arange(0.1, 0.9001, 0.0001), 4)
    gamma_values = np.round(np.arange(0.0001, 0.9001, 0.0001), 4)
    base_params = {
        'X_t': 1e6,
        'Y_t': 1e6,
        'S_t': 1,    # Current price
        'delta_t': 12/365/24/60/60,  # Time interval = 12 seconds
        'L': 1e6,       # Liquidity parameter
        'P_t': 1     # Target price
    }
    
    results_df = []
    # Run optimization for each parameter combination
    for sigma in tqdm(sigma_values):
        # Update parameters
        params = base_params.copy()
        params['sigma'] = sigma
        
        # Find optimal incoming fee rate and its revenue
        opt_gamma_in, opt_revenue_in = find_optimal_fee_in(params)
        
        # Find matching outgoing fee rate
        matching_gamma_out = find_matching_fee_out(params, opt_revenue_in, opt_gamma_in)
        
        # Add results to dataframe
        results_df.append({
            'sigma': sigma,
            'opt_gamma_in': opt_gamma_in,
            'opt_revenue': opt_revenue_in,
            'matching_gamma_out': matching_gamma_out
        })
    
    # Convert to DataFrame and save results
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('fee_comparison_results.csv', index=False)
    print("Results saved to 'fee_comparison_results.csv'")
    # results_df = pd.read_csv('fee_comparison_results.csv')
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot fee rates in the top subplot
    ax1.plot(results_df['sigma'], results_df['opt_gamma_in'], 
             color='blue', label='Optimal Incoming Fee Rate', linestyle='--')
    ax1.plot(results_df['sigma'], results_df['matching_gamma_out'], 
             color='green', label='Revenue-Matching Outgoing Fee Rate')
    ax1.set_ylabel('Fee Rate (γ)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Fee Rate Comparison')
    
    # Plot y threshold in the bottom subplot
    y_threshold = (results_df['opt_gamma_in'] - results_df['matching_gamma_out']) / \
              (results_df['matching_gamma_out'] * (1 - results_df['opt_gamma_in']))
    ax2.plot(results_df['sigma'], y_threshold, 
             color='blue', label=f'Delta y Threshold \n [{y_threshold.min():.3f}, {y_threshold.max():.3f}]')
    ax2.set_ylabel('Delta y Threshold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Target Fee Revenue from Optimal Incoming Fee Rate')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig('y_threshold_comparison.png', dpi=800, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'y_threshold_comparison.png'")

if __name__ == "__main__":
    main()