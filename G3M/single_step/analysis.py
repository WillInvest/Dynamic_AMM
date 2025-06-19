# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from g3m import calculate_G3M_expected_fee, calculate_G3M_expected_pool_value

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def G3M_analysis_with_gamma(sigma, w_values, metric='fee', L=1, pv0=1, fixed_pv=True, incoming_fee=True):
    """
    Analysis 1: How do fees scale with fee rate for different weight distributions?
    
    Args:
        gamma_fixed: Fixed gamma value for the analysis
        sigma_range: Tuple of (min_sigma, max_sigma)
        w_values: List of w values to analyze
    """

    gamma_values = np.arange(0.0005, 0.9005, 0.0005)
    
    plt.figure(figsize=(12, 8))
    
    for w in w_values:
        fees = []
        pvs = []
        tvs = []

        if fixed_pv:
            x_t = (L/((1-w)*pv0)**(1-w))**(1/w)
            y_t = (1-w) * pv0
        else:
            x_t = y_t = 1

        S_t = (w/(1-w)) * (y_t/x_t)
        initial_value = x_t * S_t + y_t
        
        for gamma in gamma_values:
            incoming_fee, outgoing_fee = calculate_G3M_expected_fee(sigma=sigma, w=w, gamma=gamma, x_t=x_t, y_t=y_t, delta_t=1)
            pv = calculate_G3M_expected_pool_value(sigma=sigma, w=w, gamma=gamma, x_t=x_t, y_t=y_t, delta_t=1)
            if incoming_fee:
                fee = incoming_fee
            else:
                fee = outgoing_fee
            
            fees.append(fee)
            pvs.append(pv)
            tvs.append(pv + fee - initial_value)
            
        if metric == 'tv':
            if w == 0.5:
                # use dashed line
                plt.plot(gamma_values, tvs, label=f'w = {w:.1f}', linewidth=2, linestyle='--')
            else:
                plt.plot(gamma_values, tvs, label=f'w = {w:.1f}', linewidth=2)
            
            plt.xlabel('Fee Rate (γ)', fontsize=12)
            plt.ylabel('Expected TV', fontsize=12)
            plt.title(f'TV Sensitivity to Fee Rate (σ = {sigma})', fontsize=14)
        elif metric == 'fee':
            if w == 0.5:
                # use dashed line
                plt.plot(gamma_values, fees, label=f'w = {w:.1f}', linewidth=2, linestyle='--')
            else:
                plt.plot(gamma_values, fees, label=f'w = {w:.1f}', linewidth=2)
            plt.xlabel('Fee Rate (γ)', fontsize=12)
            plt.ylabel('Expected Fee', fontsize=12)
            plt.title(f'Fee Sensitivity to Fee Rate (σ = {sigma})', fontsize=14)
        elif metric == 'pv':
            if w == 0.5:
                # use dashed line
                plt.plot(gamma_values, pvs, label=f'w = {w:.1f}', linewidth=2, linestyle='--')
            else:
                plt.plot(gamma_values, pvs, label=f'w = {w:.1f}', linewidth=2)
            
            plt.axhline(y=initial_value, color='black', linestyle='--', linewidth=2)
            plt.xlabel('Fee Rate (γ)', fontsize=12)
            plt.ylabel('Expected PV', fontsize=12)
            plt.title(f'PV Sensitivity to Fee Rate (σ = {sigma})', fontsize=14)
            
            
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def G3M_analysis_with_sigma(gamma, w_values, metric='fee', L=1, pv0=1, fixed_pv=True):
    """
    Analysis 1: How do fees scale with fee rate for different weight distributions?
    
    Args:
        gamma_fixed: Fixed gamma value for the analysis
        sigma_range: Tuple of (min_sigma, max_sigma)
        w_values: List of w values to analyze
    """

    sigma_values = np.arange(0.01, 1.0, 0.01)

    plt.figure(figsize=(12, 8))
    
    for w in w_values:
        fees = []
        pvs = []
        tvs = []

        if fixed_pv:
            x_t = (L/((1-w)*pv0)**(1-w))**(1/w)
            y_t = (1-w) * pv0
        else:
            x_t = y_t = 1

        S_t = (w/(1-w)) * (y_t/x_t)
        initial_value = x_t * S_t + y_t
        
        for sigma in sigma_values:
            incoming_fee, outgoing_fee = calculate_G3M_expected_fee(sigma=sigma, w=w, gamma=gamma, x_t=x_t, y_t=y_t, delta_t=1)
            if incoming_fee:
                fee = incoming_fee
            else:
                fee = outgoing_fee
            fees.append(fee)
            pv = calculate_G3M_expected_pool_value(sigma=sigma, w=w, gamma=gamma, x_t=x_t, y_t=y_t, delta_t=1)
            pvs.append(pv)
            tvs.append(pv + fee - initial_value)
            
        if metric == 'tv':
            if w == 0.5:
                # use dashed line
                plt.plot(sigma_values, tvs, label=f'w = {w:.1f}', linewidth=2, linestyle='--')
            else:
                plt.plot(sigma_values, tvs, label=f'w = {w:.1f}', linewidth=2)
            plt.xlabel('Volatility (σ)', fontsize=12)
            plt.ylabel('Expected TV', fontsize=12)
            plt.title(f'TV Sensitivity to Volatility (γ = {gamma})', fontsize=14)
        elif metric == 'fee':
            if w == 0.5:
                # use dashed line
                plt.plot(sigma_values, fees, label=f'w = {w:.1f}', linewidth=2, linestyle='--')
            else:
                plt.plot(sigma_values, fees, label=f'w = {w:.1f}', linewidth=2)
            plt.xlabel('Volatility (σ)', fontsize=12)
            plt.ylabel('Expected Fee', fontsize=12)
            plt.title(f'Fee Sensitivity to Volatility (γ = {gamma})', fontsize=14)
        elif metric == 'pv':
            if w == 0.5:
                # use dashed line
                plt.plot(sigma_values, pvs, label=f'w = {w:.1f}', linewidth=2, linestyle='--')
            else:
                plt.plot(sigma_values, pvs, label=f'w = {w:.1f}', linewidth=2)
            plt.xlabel('Volatility (σ)', fontsize=12)
            plt.ylabel('Expected PV', fontsize=12)
            plt.title(f'PV Sensitivity to Volatility (γ = {gamma})', fontsize=14)
            
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# %%

def plot_fee_difference_by_sigma_w_gamma(df):
    """
    Create 9 subplots (one for each sigma) showing outgoing_fee - incoming_fee
    vs gamma for different w values
    """

    # Calculate fee difference (outgoing - incoming)
    df['fee_difference'] = df['outgoing_fee'] - df['incoming_fee']
        
    # Get unique values
    sigma_values = sorted(df['sigma'].unique())
    w_values = sorted(df['w'].unique())
        
    print(f"Found {len(sigma_values)} sigma values: {sigma_values}")
    print(f"Found {len(w_values)} w values: {w_values}")
        
    # Create 3x3 subplot grid with more space for legend
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
        
    # Color palette for different w values
    colors = plt.cm.tab10(np.linspace(0, 1, len(w_values)))
        
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []
        
    for i, sigma in enumerate(sigma_values):
        ax = axes[i]
            
        # Filter data for this sigma
        sigma_data = df[df['sigma'] == sigma]
            
        # Plot each w value as a different line
        for j, w in enumerate(w_values):
            w_data = sigma_data[sigma_data['w'] == w].sort_values('gamma')
                
            if len(w_data) > 0:
                if w == 0.5:
                    line = ax.plot(w_data['gamma'], w_data['fee_difference'], 
                        color=colors[j], label=f'w={w:.1f}', linewidth=2, alpha=0.8, linestyle='--')
                else:
                    line = ax.plot(w_data['gamma'], w_data['fee_difference'], 
                        color=colors[j], label=f'w={w:.1f}', linewidth=2, alpha=0.8)
                
                # Collect legend info from first subplot
                if i == 0:
                    legend_handles.extend(line)
                    legend_labels.append(f'w={w:.1f}')
            
        # Formatting
        ax.set_xlabel('Fee Rate (γ)', fontsize=11)
        ax.set_ylabel('Outgoing Fee - Incoming Fee', fontsize=11)
        ax.set_title(f'σ = {sigma:.1f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    # Remove any unused subplots
    for i in range(len(sigma_values), len(axes)):
        fig.delaxes(axes[i])
    
    # Add single legend to the right of all subplots with larger font and better spacing
    fig.legend(legend_handles, legend_labels, 
              loc='center right', bbox_to_anchor=(0.96, 0.5), 
              fontsize=16, frameon=True, fancybox=True, shadow=True,
              title='Weight (w)', title_fontsize=18,
              borderpad=1.5, columnspacing=1.0, handlelength=2.5,
              handletextpad=1.0, labelspacing=1.2)
    
    # Add title with proper spacing
    fig.suptitle('Fee Difference (Outgoing - Incoming) by Volatility and Weight', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout to make room for legend and title
    plt.subplots_adjust(left=0.08, right=0.82, top=0.93, bottom=0.08, 
                       hspace=0.3, wspace=0.3)
    plt.show()
    
def plot_fee_ratio_by_sigma_w_gamma(df):
    """
    Create 9 subplots (one for each sigma) showing outgoing_fee - incoming_fee
    vs gamma for different w values
    """

    # Calculate fee difference (outgoing - incoming)
    df['fee_ratio'] = df['outgoing_fee'] / df['incoming_fee']
        
    # Get unique values
    sigma_values = sorted(df['sigma'].unique())
    w_values = sorted(df['w'].unique())
        
    print(f"Found {len(sigma_values)} sigma values: {sigma_values}")
    print(f"Found {len(w_values)} w values: {w_values}")
        
    # Create 3x3 subplot grid with more space for legend
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
        
    # Color palette for different w values
    colors = plt.cm.tab10(np.linspace(0, 1, len(w_values)))
        
    # Store legend handles and labels
    legend_handles = []
    legend_labels = []
        
    for i, sigma in enumerate(sigma_values):
        ax = axes[i]
            
        # Filter data for this sigma
        sigma_data = df[df['sigma'] == sigma]
            
        # Plot each w value as a different line
        for j, w in enumerate(w_values):
            w_data = sigma_data[sigma_data['w'] == w].sort_values('gamma')
                
            if len(w_data) > 0:
                if w == 0.5:
                    line = ax.plot(w_data['gamma'], w_data['fee_ratio'], 
                        color=colors[j], label=f'w={w:.1f}', linewidth=2, alpha=0.8, linestyle='--')
                else:
                    line = ax.plot(w_data['gamma'], w_data['fee_ratio'], 
                        color=colors[j], label=f'w={w:.1f}', linewidth=2, alpha=0.8)
                
                # Collect legend info from first subplot
                if i == 0:
                    legend_handles.extend(line)
                    legend_labels.append(f'w={w:.1f}')

        # Formatting
        ax.set_xlabel('Fee Rate (γ)', fontsize=11)
        ax.set_ylabel('Outgoing Fee / Incoming Fee', fontsize=11)
        ax.set_title(f'σ = {sigma:.1f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    # Remove any unused subplots
    for i in range(len(sigma_values), len(axes)):
        fig.delaxes(axes[i])
    
    # Add single legend to the right of all subplots with larger font and better spacing
    fig.legend(legend_handles, legend_labels, 
              loc='center right', bbox_to_anchor=(0.96, 0.5), 
              fontsize=16, frameon=True, fancybox=True, shadow=True,
              title='Weight (w)', title_fontsize=18,
              borderpad=1.5, columnspacing=1.0, handlelength=2.5,
              handletextpad=1.0, labelspacing=1.2)
    
    # Add title with proper spacing
    fig.suptitle('Fee Ratio (Outgoing / Incoming) by Volatility and Weight', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout to make room for legend and title
    plt.subplots_adjust(left=0.08, right=0.82, top=0.93, bottom=0.08, 
                       hspace=0.3, wspace=0.3)
    plt.show()

# %%
import pandas as pd
from tqdm import tqdm

sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gammas = np.round(np.arange(0.001, 0.5, 0.001), 3)
w_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
delta_t = 1
pv_0 = 1
L = 1
results = []
total_comb = len(sigmas) * len(w_values) * len(gammas)
progress_bar = tqdm(total=total_comb, desc="Processing combinations")

for sigma in sigmas:
    for w in w_values:
        for gamma in gammas:
            incoming_fee, outgoing_fee = calculate_G3M_expected_fee(sigma, w, gamma, pv_0, L, delta_t)
            results.append({
                'sigma': sigma,
                'w': w,
                'gamma': gamma,
                'incoming_fee': incoming_fee,
                'outgoing_fee': outgoing_fee
            })
            progress_bar.update(1)

progress_bar.close()

df = pd.DataFrame(results)
plot_fee_difference_by_sigma_w_gamma(df)
plot_fee_ratio_by_sigma_w_gamma(df)

# %%


# plot fee with fix sigma
print("--------------------------------------------------")
print("plot with fixed sigma and initial token reserves")
print("--------------------------------------------------")
G3M_analysis_with_gamma(sigma=0.3, w_values=[0.4, 0.6], metric='fee', fixed_pv=False, incoming_fee=False)
G3M_analysis_with_gamma(sigma=0.3, w_values=[0.4, 0.6], metric='pv', fixed_pv=False, incoming_fee=False)
G3M_analysis_with_gamma(sigma=0.3, w_values=[0.4, 0.6], metric='tv', fixed_pv=False, incoming_fee=False)

# %%


# plot fee with fix sigma
print("--------------------------------------------------")
print("plot with fixed sigma and initial pool value")
print("--------------------------------------------------")
G3M_analysis_with_gamma(sigma=0.3, w_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric='fee', fixed_pv=True, incoming_fee=False)
G3M_analysis_with_gamma(sigma=0.3, w_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric='pv', fixed_pv=True, incoming_fee=False)
G3M_analysis_with_gamma(sigma=0.3, w_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric='tv', fixed_pv=True, incoming_fee=False)

# %%

# # plot fee with fix gamma
# print("--------------------------------------------------")
# print("plot with fixed gamma and initial token reserves")
# print("--------------------------------------------------")
# G3M_analysis_with_sigma(gamma=0.003, w_values=[0.4, 0.6], metric='fee', fixed_pv=False)
# G3M_analysis_with_sigma(gamma=0.003, w_values=[0.4, 0.6], metric='pv', fixed_pv=False)
# G3M_analysis_with_sigma(gamma=0.003, w_values=[0.4, 0.6], metric='tv', fixed_pv=False)


# %%

# plot fee with fix gamma
print("--------------------------------------------------")
print("plot with fixed gamma and initial pool value")
print("--------------------------------------------------")
G3M_analysis_with_sigma(gamma=0.003, w_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric='fee', fixed_pv=True)
G3M_analysis_with_sigma(gamma=0.003, w_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric='pv', fixed_pv=True)
G3M_analysis_with_sigma(gamma=0.003, w_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric='tv', fixed_pv=True)

# %%

