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
    S_t = params['S_t']
    X_t = params['X_t']
    Y_t = params['Y_t']
    sigma = params['sigma']
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
    sigma = params['sigma']
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

def find_matching_fee_out(params, target_revenue, gamma_in):
    """Find the outgoing fee rate that achieves the target revenue."""
    def objective(gamma):
        if gamma <= 0 or gamma >= 1:
            return np.inf
        fee_revenue = calculate_fee_out(gamma, params)
        return (fee_revenue - target_revenue) ** 2
    
    result = optimize.minimize_scalar(objective, bounds=(0.0001, gamma_in), method='bounded')
    return result.x

def main():
    # Define parameter grid
    sigma_values = np.round(np.arange(0.01, 1.01, 0.01), 2)  # Coarser grid for sigma
    gamma_in_values = np.round(np.arange(0.0001, 0.9001, 0.0001), 4)  # Coarser grid for gamma_in
    
    # base_params = {
    #     'X_t': 1,
    #     'Y_t': 1,
    #     'S_t': 1,    # Current price
    #     'delta_t': 1,  # Time interval
    #     'L': 1,       # Liquidity parameter
    #     'P_t': 1     # Target price
    # }
    
    # results = []
    # total_combinations = len(sigma_values) * len(gamma_in_values)
    
    # with tqdm(total=total_combinations) as pbar:
    #     for sigma in sigma_values:
    #         params = base_params.copy()
    #         params['sigma'] = sigma
            
    #         for gamma_in in gamma_in_values:
    #             # Calculate incoming fee revenue
    #             fee_revenue_in = calculate_fee_in(gamma_in, params)
                
    #             # Find matching outgoing fee rate
    #             gamma_out = find_matching_fee_out(params, fee_revenue_in, gamma_in)
                
    #             # Calculate delta y threshold
    #             delta_y = (gamma_in - gamma_out) / (gamma_out * (1 - gamma_in))
                
    #             results.append({
    #                 'sigma': sigma,
    #                 'gamma_in': gamma_in,
    #                 'fee_revenue': fee_revenue_in,
    #                 'gamma_out': gamma_out,
    #                 'delta_y': delta_y
    #             })
                
    #             pbar.update(1)
    
    # # Convert to DataFrame and save results
    # results_df = pd.DataFrame(results)
    # results_df.to_csv('fee_grid_comparison_results.csv', index=False)
    # print("Results saved to 'fee_grid_comparison_results.csv'")
    results_df = pd.read_csv('fee_grid_comparison_results.csv')
    results_df = results_df[results_df['sigma'] == 0.1]
    # Create heatmap of delta y threshold
    plt.figure(figsize=(10, 8))
    pivot_df = results_df.pivot(index='sigma', columns='gamma_in', values='delta_y')
    plt.imshow(pivot_df, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Delta y Threshold')
    
    # Set ticks and labels
    plt.xticks(range(len(gamma_in_values)), [f'{g:.1f}' for g in gamma_in_values])
    plt.yticks(range(len(sigma_values)), [f'{s:.1f}' for s in sigma_values])
    
    plt.xlabel('Incoming Fee Rate (γ_in)')
    plt.ylabel('Sigma (σ)')
    plt.title('Delta y Threshold Heatmap')
    
    plt.savefig('fee_grid_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Heatmap saved as 'fee_grid_comparison_heatmap.png'")

if __name__ == "__main__":
    main() 