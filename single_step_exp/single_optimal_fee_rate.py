import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt

def fee_in_optimization(params):
    """Find optimal gamma to maximize the incoming fee using two methods.
    
    Parameters:
    params: dict containing S_t, X_t, Y_t, sigma, delta_t, L, P_t
    
    Returns:
    dict with optimization results from both methods
    """
    S_t = params['S_t']
    X_t = params['X_t']
    Y_t = params['Y_t']
    sigma = params['sigma']
    delta_t = params['delta_t']
    L = params['L']
    P_t = params['P_t']
    
    def alpha(gamma):
        """Calculate alpha based on gamma."""
        return L * np.sqrt((1 - gamma) * S_t * np.exp(-0.25 * sigma**2 * delta_t))
    
    def d1(gamma):
        """Calculate d1 based on gamma."""
        return np.log((1 - gamma) * P_t / S_t) / (sigma * np.sqrt(delta_t))
    
    def d2(gamma):
        """Calculate d2 based on gamma."""
        return np.log(P_t / ((1 - gamma) * S_t)) / (sigma * np.sqrt(delta_t))
    
    def d1_minus(gamma):
        """Calculate d1- based on gamma."""
        return d1(gamma) - 0.5 * sigma * np.sqrt(delta_t)
    
    def d1_plus(gamma):
        """Calculate d1+ based on gamma."""
        return d1(gamma) + 0.5 * sigma * np.sqrt(delta_t)
    
    def d2_plus(gamma):
        """Calculate d2+ based on gamma."""
        return d2(gamma) + 0.5 * sigma * np.sqrt(delta_t)
    
    def d2_minus(gamma):
        """Calculate d2- based on gamma."""
        return d2(gamma) - 0.5 * sigma * np.sqrt(delta_t)
    
    def Fin(gamma):
        """Calculate F(gamma) - the term inside parentheses in Fee_in."""
        a = alpha(gamma)

        term1 = a * (norm.cdf(d1(gamma)) + norm.cdf(-d2(gamma)))
        term2 = S_t * X_t * norm.cdf(d1_minus(gamma))
        term3 = Y_t * norm.cdf(-d2_plus(gamma))
        return term1 - term2 - term3
    
    def Fout(gamma):
        """Calculate F(gamma) - the term inside parentheses in Fee_out."""
        a = alpha(gamma)
        term1 = a/(1-gamma) * (norm.cdf(d1(gamma)) + norm.cdf(-d2(gamma)))
        term2 = S_t * X_t * norm.cdf(-d2_minus(gamma))
        term3 = Y_t * norm.cdf(d1_plus(gamma))
        return -term1 + term2 + term3
    
    def dFin(gamma):
        """Calculate derivative of F with respect to gamma."""
        a = alpha(gamma)
        
        # Normal PDFs (φ)
        phi_d1 = norm.pdf(d1(gamma))
        phi_d2 = norm.pdf(d2(gamma))
        phi_d1_minus = norm.pdf(d1_minus(gamma))
        phi_d2_plus = norm.pdf(d2_plus(gamma))
        
        # Normal CDFs (Φ)
        Phi_d1 = norm.cdf(d1(gamma))
        Phi_minus_d2 = norm.cdf(-d2(gamma))
        
        # da/dgamma
        da_dgamma = -a / (2 * (1 - gamma))
        
        # dd1/dgamma and dd2/dgamma
        dd1_dgamma = -1 / ((1 - gamma) * sigma * np.sqrt(delta_t))
        dd2_dgamma = 1 / ((1 - gamma) * sigma * np.sqrt(delta_t))
        
        # First term: da/dgamma * (Φ(d1) + Φ(-d2))
        term1 = da_dgamma * (Phi_d1 + Phi_minus_d2)
        
        # Second term: a * (φ(d1) * dd1/dgamma - φ(d2) * dd2/dgamma)
        term2 = a * (phi_d1 * dd1_dgamma - phi_d2 * dd2_dgamma)
        
        # Third term: -S_t * X_t * φ(d1-) * dd1/dgamma
        term3 = -S_t * X_t * phi_d1_minus * dd1_dgamma
        
        # Fourth term: -Y_t * (-φ(d2+)) * dd2/dgamma
        term4 = Y_t * phi_d2_plus * dd2_dgamma
        
        return term1 + term2 + term3 + term4
    
    def fee_out(gamma):
        """Calculate Fee_out for a given gamma."""
        return gamma * Fout(gamma)
    
    def fee_in(gamma):
        """Calculate Fee_in for a given gamma."""
        return gamma / (1 - gamma) * Fin(gamma)
    
    def objective_in(gamma):
        """Function to minimize - negative of Fee_in."""
        if gamma <= 0 or gamma >= 1:
            return np.inf  # Invalid range
        return -fee_in(gamma)
    
    def objective_out(gamma):
        """Function to minimize - negative of Fee_out."""
        if gamma <= 0 or gamma >= 1:
            return np.inf  # Invalid range
        return -fee_out(gamma)
    
    def optimization_equation(gamma):
        """Equation that should be zero at the optimal gamma."""
        if gamma <= 0 or gamma >= 1:
            return np.inf  # Invalid range
        return Fin(gamma) + gamma * (1 - gamma) * dFin(gamma)
    
    # Method 1: Direct optimization of Fee_in
    result_in = optimize.minimize_scalar(objective_in, bounds=(0.0001, 0.9999), method='bounded')
    gamma_opt_in = result_in.x
    max_fee_in = -result_in.fun
    
    result_out = optimize.minimize_scalar(objective_out, bounds=(0.0001, 0.9999), method='bounded')
    gamma_opt_out = result_out.x
    max_fee_out = -result_out.fun
    
    # # Method 2: Find root of the optimization equation
    # try:
    #     gamma_opt2 = optimize.brentq(optimization_equation, 0.0001, 0.9999)
    #     max_fee2 = fee_in(gamma_opt2)
    # except:
    #     gamma_opt2 = None
    #     max_fee2 = -np.inf
    
    # Return results from both methods
    return {
        'opt_gamma_in': gamma_opt_in,
        'opt_fee_in': max_fee_in,
        'opt_gamma_out': gamma_opt_out,
        'opt_fee_out': max_fee_out
    }
    
    
if __name__ == "__main__":
    from tqdm import tqdm
    results_df = []
    # Define parameter ranges to test
    sigma_values = np.round(np.arange(0.1, 2.0001, 0.0001), 4)
    base_params = {
        'X_t': 1000,
        'Y_t': 1000,
        'S_t': 1,    # Current price
        'delta_t': 1,  # Time interval
        'L': 1000,       # Liquidity parameter
        'P_t': 1     # Target price
    }
    
    # Run optimization for each parameter combination
    for sigma in tqdm(sigma_values):
        # Update parameters
        params = base_params.copy()
        params['sigma'] = sigma
        
        # Run optimization
        opt_results = fee_in_optimization(params)
        
        # Add results to dataframe
        results_df.append({
            'sigma': sigma,
            'opt_gamma_in': opt_results['opt_gamma_in'],
            'opt_fee_in': opt_results['opt_fee_in'],
            'opt_gamma_out': opt_results['opt_gamma_out'],
            'opt_fee_out': opt_results['opt_fee_out']
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results_df)
    results_df.to_csv('fee_optimization_results.csv', index=False)
    print("Results saved to 'fee_optimization_results.csv'")
    
    # Create the dual y-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot optimal gamma on left y-axis
    ax1.plot(results_df['sigma'], results_df['opt_gamma_in'], color='blue', label='Optimal Fee In')
    ax1.plot(results_df['sigma'], results_df['opt_gamma_out'], color='green', label='Optimal Fee Out')
    ax1.set_xlabel('Sigma (σ)')
    ax1.set_ylabel('Optimal Fee (γ)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis for fee
    ax2 = ax1.twinx()
    ax2.plot(results_df['sigma'], results_df['opt_fee_in'], color='red', label='Fee Revenue In')
    ax2.plot(results_df['sigma'], results_df['opt_fee_out'], color='orange', label='Fee Revenue Out')
    ax2.set_ylabel('Fee Revenue', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add title and legend
    plt.title('Optimal Gamma and Fee Revenue vs Sigma')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('gamma_fee_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as 'gamma_fee_plot.png'")
  