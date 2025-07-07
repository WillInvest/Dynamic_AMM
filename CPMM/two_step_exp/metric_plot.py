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

def subplot_comparison(path: str, metric: str):
    """
    Create multiple comparison plots from parquet data showing step and fee source comparisons
    
    Parameters:
    -----------
    path : str
        Path to the parquet file
    metric : str
        Metric to plot
    """
    df = pl.read_parquet(path)
    df = df.filter(pl.col('metric') == metric)
    
    # Get all sigma values and sort them
    subplot_values = df.get_column('sigma').unique().sort()
    x_column = 'fee_rate'
    x_label = 'Fee Rate (bps)'
    x_scale = 10000  # Convert to bps

    # Line styles for each combination
    styles = {
        (1, 'in'): ('black', '-', 'Step 1 incoming'),
        (1, 'out'): ('red', '-', 'Step 1 outgoing'),
        (2, 'in'): ('black', '--', 'Step 2 incoming'),
        (2, 'out'): ('red', '--', 'Step 2 outgoing')
    }
    
    # Calculate number of figures needed for 5x5 grid
    n_total_values = len(subplot_values)
    n_figures = ceil(n_total_values / 9)  # 25 = 5x5 grid
    
    # Create each figure
    for fig_idx in range(n_figures):
        # Get values for this figure
        start_idx = fig_idx * 9  # Changed from 25 to 9 for 3x3 grid
        end_idx = min((fig_idx + 1) * 9, len(subplot_values))
        values_for_fig = subplot_values[start_idx:end_idx]
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 8))  # Changed from 5,5 to 3,3
        axes = axes.flatten()
        
        # Plot each value in this figure
        for subplot_idx, value in enumerate(values_for_fig):
            ax = axes[subplot_idx]
            subplot_data = df.filter(pl.col('sigma') == value)
            
            # Plot each combination of step and fee_source
            for (step, fee_source), (color, linestyle, label) in styles.items():
                line_data = subplot_data.filter(
                    (pl.col('step') == step) & 
                    (pl.col('fee_source') == fee_source)
                ).sort(x_column)
                
                # Convert to numpy arrays for plotting
                x_values = line_data.get_column(x_column).to_numpy() * x_scale
                y_values = line_data.get_column('value').to_numpy()
                
                # Plot line
                if len(y_values) > 0:
                    ax.plot(x_values, y_values, color=color, linestyle=linestyle, 
                           label=label if subplot_idx == 0 else "")
                    
                    # Find and plot optimal point
                    optimal_idx = np.argmax(y_values)
                    optimal_x = x_values[optimal_idx]
                    optimal_y = y_values[optimal_idx]
                    ax.scatter(optimal_x, optimal_y, color=color, marker='o', s=20, zorder=5)
                    ax.axvline(x=optimal_x, color=color, linestyle=':', alpha=0.3)
            
            ax.set_title(f'σ = {value:.2f}')
            ax.set_xlabel(x_label if subplot_idx >= 6 else '')  # Changed from 20 to 6 for bottom row
            ax.set_ylabel(f'{metric} value' if subplot_idx % 3 == 0 else '')  # Changed from 5 to 3 for left column
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(values_for_fig), len(axes)):
            fig.delaxes(axes[idx])
        
        # Create suptitle and adjust spacing for legend
        suptitle = f'{metric.title()} Comparison'
        fig.suptitle(suptitle, fontsize=16, y=0.95)
        
        # Create legend at the top of the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92),
                  ncol=4)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.84)
        plt.show()

def plot_value_and_vega_comparison(path: str, metric: str) -> None:
    """
    Plot value and vega comparison for a given metric
    
    Args:
        path: Path to the parquet file containing metrics data
        metric: Name of the metric to plot
    """
    # Set default value
    xaxis = 'sigma'
    
    df = pl.read_parquet(path)
    df = df.filter(pl.col('metric') == metric)
    
    # Create 2x2 subplots
    plt.figure(figsize=(12, 8))
    
    # Get unique values for the color mapping
    if xaxis == 'sigma':
        color_param = 'fee_rate'
        color_label = 'Fee Rate'
        x_label = 'σ (Sigma)'
    else:  # xaxis == 'fee_rate'
        color_param = 'sigma'
        color_label = 'Sigma'
        x_label = 'Fee Rate'
    
    unique_colors = df[color_param].unique().sort()
    
    # Create colormap
    cmap = plt.colormaps['viridis']
    norm = Normalize(vmin=unique_colors.min(), vmax=unique_colors.max())
    
    # Iterate through fee sources
    for idx, fee_source in enumerate(['incoing', 'outgoing']):
        drift = False
        
        # Plot for each color parameter value
        for color_value in unique_colors:
            rate_data = df.filter(
                (pl.col('fee_source') == fee_source) & 
                (pl.col('drift') == drift) &
                (pl.col(color_param) == color_value)
            )
            
            x_values = rate_data[xaxis].to_numpy()
            values = rate_data['value'].to_numpy()
            vegas = rate_data['vega'].to_numpy()
            
            if len(x_values) > 0:
                sort_idx = np.argsort(x_values)
                x_values = x_values[sort_idx]
                values = values[sort_idx]
                vegas = vegas[sort_idx]
                
                # Plot value
                plt.subplot(2, 2, idx * 2 + 1)
                plt.plot(x_values, values, 
                        color=cmap(norm(color_value)), 
                        label=f'{color_label} = {color_value:.4f}')
                
                # Plot vega
                plt.subplot(2, 2, idx * 2 + 2)
                plt.plot(x_values, vegas, 
                        color=cmap(norm(color_value)), 
                        label=f'{color_label} = {color_value:.4f}')
        
        # Set titles and labels for value subplot
        plt.subplot(2, 2, idx * 2 + 1)
        plt.title(f'{fee_source.capitalize()} No Drift - Value')
        plt.xlabel(x_label)
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set titles and labels for vega subplot
        plt.subplot(2, 2, idx * 2 + 2)
        plt.title(f'{fee_source.capitalize()} No Drift - Vega')
        plt.xlabel(x_label)
        plt.ylabel('Vega')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and add colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(color_label, rotation=270, labelpad=15)
    
    plt.show()
