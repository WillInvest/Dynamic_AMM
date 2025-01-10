from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Union, TypedDict
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
from math import ceil
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def subplot_comparison(path: str,
                        metric: str = 'expected_fee',
                        value_type: str = 'value',
                        drift: bool = None,
                        max_plots: int = None,
                        max_sigma: float = None,
                        plot_rows: int = 5,
                        subplot_by: str = 'sigma'):
    """
    Create multiple comparison plots from parquet data with optimal points and difference bars
    
    Parameters:
    -----------
    path : str
        Path to the parquet file
    metric : str, optional
        Metric to plot (default: 'expected_fee')
    value_type : str, optional
        Value type to plot (default: 'value')
    drift : bool, optional
        Filter for drift condition. 
        - None (default): plot all conditions
        - True: only with drift
        - False: only without drift
    max_plots : int, optional
        Maximum number of plots to generate (default: None, which plots all)
    max_sigma : float, optional
        Maximum sigma value to include in plots
    subplot_by : str, optional
        What parameter to use for subplots ('sigma' or 'fee_rate', default: 'sigma')
    """
    # Load data
    df = pl.read_parquet(path)
    df = df.filter(pl.col('metric') == metric)
    
    # Additional drift filtering if specified
    if drift is not None:
        df = df.filter(pl.col('drift') == drift)
        
    # Filter by max_sigma if specified
    if max_sigma is not None:
        df = df.filter(pl.col('sigma') <= max_sigma)
        
    # Get all values to subplot by and sort them
    if subplot_by == 'sigma':
        subplot_values = df.get_column('sigma').unique().sort()
        x_column = 'fee_rate'
        x_label = 'Fee Rate (bps)'
        x_scale = 10000  # Convert to bps
    elif subplot_by == 'fee_rate':
        subplot_values = df.get_column('fee_rate').unique().sort()
        x_column = 'sigma'
        x_label = 'σ (Sigma)'
        x_scale = 1
    else:
        raise ValueError("subplot_by must be either 'sigma' or 'fee_rate'")

    n_total_values = len(subplot_values)
    
    # Determine number of plots
    subplots_per_fig = plot_rows * plot_rows
    if max_plots is not None:
        n_figures = min(max_plots, ceil(n_total_values / subplots_per_fig))
    else:
        n_figures = ceil(n_total_values / subplots_per_fig)
    
    # Select values to plot
    if max_plots is not None and n_total_values > max_plots * subplots_per_fig:
        # If more values than can be plotted, select evenly spaced values
        value_indices = np.linspace(0, n_total_values - 1, max_plots * subplots_per_fig, dtype=int)
        selected_values = subplot_values[value_indices]
    else:
        selected_values = subplot_values
    
    # Line styles
    styles = {
        (True, True): ('blue', '--', 'Ingoing with drift'),
        (True, False): ('blue', '-', 'Ingoing no drift'),
        (False, True): ('red', '--', 'Outgoing with drift'),
        (False, False): ('red', '-', 'Outgoing no drift')
    }
    
    # Create each figure
    for fig_idx in range(n_figures):
        # Get values for this figure
        start_idx = fig_idx * subplots_per_fig
        end_idx = min((fig_idx + 1) * subplots_per_fig, len(selected_values))
        values_for_fig = selected_values[start_idx:end_idx]
        fig, axes = plt.subplots(plot_rows, plot_rows, figsize=(12, 8))
        axes = axes.flatten()
        
        # Plot each value in this figure
        for subplot_idx, value in enumerate(values_for_fig):
            ax = axes[subplot_idx]
            subplot_data = df.filter(pl.col(subplot_by) == value)
            
            # Create second y-axis for difference bars
            ax2 = ax.twinx()
            
            # Dictionary to store values for difference calculation
            line_values = {}
            
            # Plot each combination
            for (is_ingoing, has_drift), (color, linestyle, label) in styles.items():
                # Skip combinations not matching drift filter if specified
                if drift is not None and drift != has_drift:
                    continue
                
                line_data = subplot_data.filter(
                    (pl.col('fee_source') == ('ingoing' if is_ingoing else 'outgoing')) &
                    (pl.col('drift') == has_drift)
                ).sort(x_column)
        
                # Convert to numpy arrays for plotting
                x_values = line_data.get_column(x_column).to_numpy() * x_scale
                y_values = line_data.get_column(value_type).to_numpy()
                
                # Store values for difference calculation
                line_values[is_ingoing] = y_values
        
                # Plot line with updated label
                ax.plot(x_values, y_values, color=color, linestyle=linestyle, label=label)
        
                # Find and plot optimal point
                if len(y_values) > 0:
                    optimal_idx = np.argmax(y_values)
                    optimal_x = x_values[optimal_idx]
                    optimal_y = y_values[optimal_idx]
                    ax.scatter(optimal_x, optimal_y, color=color, marker='o', s=20, zorder=5)
                    ax.axvline(x=optimal_x, color=color, linestyle=':', alpha=0.3)
            
            # Calculate and plot difference bars if we have both ingoing and outgoing values
            if True in line_values and False in line_values:
                diff = line_values[False] - line_values[True]  # outgoing - ingoing
                ax2.bar(x_values, diff, alpha=0.2, color='gray', width=x_values[1]-x_values[0] if len(x_values) > 1 else 0.1)
                ax2.set_ylabel('Difference (Outgoing - Ingoing)', color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
            
            if subplot_by == 'sigma':
                ax.set_title(f'σ = {value:.2f}')
            else:
                ax.set_title(f'Fee Rate = {value*10000:.0f} bps')
            ax.set_xlabel(x_label if subplot_idx >= 20 else '')
            ax.set_ylabel(f'{metric} {value_type}' if subplot_idx % 5 == 0 else '')
            ax.tick_params(labelsize=8)
            ax.grid(False)
            ax.legend(loc='upper right', fontsize=8)
        
        # Create suptitle with larger font size and adjust vertical position
        drift_text = 'With Drift' if drift is True else 'No Drift' if drift is False else 'All Conditions'
        subplot_text = 'By Sigma' if subplot_by == 'sigma' else 'By Fee Rate'
        suptitle = f'{metric} {value_type} Comparison - {subplot_text} ({drift_text})'
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout()
        plt.show()
    

def plot_value_and_vega_comparison(path: str, metric: str) -> None:
   df = pl.read_parquet(path)
   df = df.filter(pl.col('metric') == metric)
   unique_fee_rates = df['fee_rate'].unique().sort()
   
   # Create 2x2 subplots
   plt.figure(figsize=(12, 8))
   
   # Create colormap
   cmap = plt.colormaps['viridis']
   norm = Normalize(vmin=unique_fee_rates.min(), vmax=unique_fee_rates.max())
   
   # Iterate through fee sources
   for idx, fee_source in enumerate(['ingoing', 'outgoing']):
       drift = False
       
       # Plot for each fee rate
       for fee_rate in unique_fee_rates:
           rate_data = df.filter(
               (pl.col('fee_source') == fee_source) & 
               (pl.col('drift') == drift) &
               (pl.col('fee_rate') == fee_rate)
           )
           
           sigmas = rate_data['sigma'].to_numpy()
           values = rate_data['value'].to_numpy()
           vegas = rate_data['vega'].to_numpy()
           
           if len(sigmas) > 0:
               sort_idx = np.argsort(sigmas)
               sigmas = sigmas[sort_idx]
               values = values[sort_idx]
               vegas = vegas[sort_idx]
               
               # Plot value
               plt.subplot(2, 2, idx * 2 + 1)
               plt.plot(sigmas, values, 
                       color=cmap(norm(fee_rate)), 
                       label=f'Fee Rate = {fee_rate:.4f}')
               
               # Plot vega
               plt.subplot(2, 2, idx * 2 + 2)
               plt.plot(sigmas, vegas, 
                       color=cmap(norm(fee_rate)), 
                       label=f'Fee Rate = {fee_rate:.4f}')
       
       # Set titles and labels for value subplot
       plt.subplot(2, 2, idx * 2 + 1)
       plt.title(f'{fee_source.capitalize()} No Drift - Value')
       plt.xlabel('σ (Sigma)')
       plt.ylabel('Value')
       plt.grid(True, linestyle='--', alpha=0.7)
       
       # Set titles and labels for vega subplot
       plt.subplot(2, 2, idx * 2 + 2)
       plt.title(f'{fee_source.capitalize()} No Drift - Vega')
       plt.xlabel('σ (Sigma)')
       plt.ylabel('Vega')
       plt.grid(True, linestyle='--', alpha=0.7)
   
   # Adjust layout and add colorbar
   plt.tight_layout(rect=[0, 0, 0.9, 1])
   cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
   sm = ScalarMappable(cmap=cmap, norm=norm)
   sm.set_array([])
   cbar = plt.colorbar(sm, cax=cbar_ax)
   cbar.set_label('Fee Rate', rotation=270, labelpad=15)
   
   plt.show()
