# Import required libraries
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class MetricAnalyzer:
    def __init__(self, path: str, metric_col: str):
        """
        Initialize MetricAnalyzer with data path and metric column.
        
        Args:
            path: Path to parquet file
            metric_col: 
            [
                'total_fee_dollar_value',
                'trader_total_pnl',
                'impermanent_loss',
                'net_profit',
                'total_number_trade',
                'account_profit'
            ]
        """
        self.path = path
        self.metric_col = metric_col
        self.formatted_name = metric_col.replace('_', ' ').title()
        self.lazy_df = (pl.scan_parquet(path)
            .with_columns([
                pl.col('sigma').round(3),
                pl.col('fee_rate').round(3)
            ]))
        # For fee rates
        self.fee_rates = (self.lazy_df
                          .select('fee_rate')
                          .unique()
                          .collect(streaming=True)  
                          .sort('fee_rate')  # Sort before converting to list
                          ['fee_rate']
                          .to_list())

        # For sigmas
        self.sigmas = (self.lazy_df
                       .select('sigma')
                       .unique()
                       .collect(streaming=True)
                       .sort('sigma')    # Sort before converting to list
                       ['sigma']
                       .to_list())
        
    def get_unique_sigmas(self, exclude_last: bool = True):
        """Get unique sigma values from the data."""
        sigmas = (self.lazy_df
                 .select('sigma')
                 .unique()
                 .collect(streaming=True)
                 .sort('sigma')['sigma']
                 .to_list())
        
        return sigmas[:-1] if exclude_last else sigmas
    
    def get_stats(self, sigma: float, CI_type: str = 'IQR'):
        """Get statistics for a specific sigma value."""
        if CI_type == 'IQR':
            stats = (self.lazy_df
                    .filter(pl.col('sigma') == sigma)
                    .group_by('fee_rate')
                    .agg([
                        pl.col(self.metric_col).mean().alias('mean'),
                        pl.col(self.metric_col).quantile(0.25).alias('lower'),
                        pl.col(self.metric_col).quantile(0.75).alias('upper')
                    ])
                    .collect(streaming=True)
                    .sort('fee_rate'))
        elif CI_type == 'STD':
            stats = (self.lazy_df
                    .filter(pl.col('sigma') == sigma)
                    .group_by('fee_rate')
                    .agg([
                        pl.col(self.metric_col).mean().alias('mean'),
                        (pl.col(self.metric_col).mean() - pl.col(self.metric_col).std()).alias('lower'),
                        (pl.col(self.metric_col).mean() + pl.col(self.metric_col).std()).alias('upper')
                    ])
                    .collect(streaming=True)
                    .sort('fee_rate'))
        elif CI_type == 'CLT':
            stats = (self.lazy_df
                    .filter(pl.col('sigma') == sigma)
                    .group_by('fee_rate')
                    .agg([
                        pl.col(self.metric_col).mean().alias('mean'),
                        (pl.col(self.metric_col).mean() - 
                         (pl.col(self.metric_col).std() / pl.count().sqrt())).alias('lower'),
                        (pl.col(self.metric_col).mean() + 
                         (pl.col(self.metric_col).std() / pl.count().sqrt())).alias('upper')
                    ])
                    .collect(streaming=True)
                    .sort('fee_rate'))
        return stats

    def plot_line(self, 
                  exclude_last_sigma: bool = True,
                  n_cols: int = 3,
                  color: str = 'blue',
                  CI: bool = True,
                  CI_type: str = 'IQR',
                  figsize_per_subplot: tuple = (5, 4)):
        """
        Create line plots with confidence intervals.
        CI_type = ['IQR', 'STD', 'CLT']
        """
        unique_sigmas = self.get_unique_sigmas(exclude_last_sigma)
        
        n_plots = len(unique_sigmas)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(figsize_per_subplot[0]*n_cols, 
                                       figsize_per_subplot[1]*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        ci_label = {'IQR': 'IQR', 'STD': '±1 STD', 'CLT': '±1 SEM'}[CI_type]

        for idx, sigma in enumerate(unique_sigmas):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            stats = self.get_stats(sigma, CI_type)
            
            # Plot mean line
            ax.plot(stats['fee_rate'], stats['mean'], color=color, label='Mean')
            
            # Plot confidence interval
            if CI:
                ax.fill_between(stats['fee_rate'], 
                              stats['lower'], 
                              stats['upper'], 
                              alpha=0.2, 
                              color=color,
                              label=ci_label)
            
            self._customize_subplot(ax, sigma, idx, col)
        
        self._finalize_plot(fig, axes, n_rows, n_cols, len(unique_sigmas), CI_type)
        return fig
    
    def _customize_subplot(self, ax, sigma, idx, col):
        """Helper method to customize individual subplots."""
        ax.set_title(f'σ = {sigma}')
        ax.set_xlabel('Fee Rate')
        if col == 0:
            ax.set_ylabel(self.formatted_name)
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _finalize_plot(self, fig, axes, n_rows, n_cols, n_plots, CI_type=None):
        """Helper method to finalize the plot."""
        # Remove empty subplots
        for idx in range(n_plots, n_rows * n_cols):
            fig.delaxes(axes[idx // n_cols, idx % n_cols])
    
        # Generate title with CI_type
        title = f'{self.formatted_name} Analysis by Sigma'
        if CI_type:
            ci_type_map = {
                'IQR': 'Interquartile Range (IQR)',
                'STD': '±1 Standard Deviation',
                'CLT': '±1 Standard Error of the Mean (SEM)'
            }
            ci_type_text = ci_type_map.get(CI_type, CI_type)
            title += f' ({ci_type_text})'
    
        # Add main title
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
        # Adjust layout and spacing
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    
    def plot_distributions(self, sigma: float = 0.5, 
                                 figsize: tuple = (15, 15)):
        """
        Create distribution plots of the metric for a specific fee rate across all sigma values.
    
        Args:
            fee_rate: The specific fee rate to analyze
            figsize: Figure size tuple (width, height)
        """
        all_fee_rates = sorted(self.fee_rates)
        fee_rate_idx = np.linspace(0, len(all_fee_rates) - 1, 9, dtype=int)
        fee_rates = [all_fee_rates[i] for i in fee_rate_idx]

        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()  # Flatten axes for easy iteration

        # Process each sigma and fee_rate combination
        for idx, fee_rate in enumerate(fee_rates):
            ax = axes[idx]
    
            # Get data for this sigma and fee_rate
            data = (self.lazy_df
                    .filter((pl.col('sigma') == sigma) & (pl.col('fee_rate') == fee_rate))
                    .select(self.metric_col)
                    .collect(streaming=True))
        
            # Create distribution plot if data is not empty
            if data.height > 0:
                sns.histplot(data=data, x=self.metric_col, ax=ax, kde=True)
            
                # Calculate and add summary statistics
                mean_val = float(data[self.metric_col].mean())
                median_val = float(data[self.metric_col].median())
            
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label='Mean')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label='Median')

                # Add summary statistics text
                stats_text = (f'Mean: {mean_val:.2e}\n'
                             f'Median: {median_val:.2e}\n'
                             f'Std: {float(data[self.metric_col].std()):.2e}')
                ax.text(0.95, 0.95, stats_text, 
                        transform=ax.transAxes, 
                        verticalalignment='top', 
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # If no data, display a placeholder
                ax.text(0.5, 0.5, 'No Data', 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        transform=ax.transAxes, 
                        fontsize=12, color='gray')

            # Customize subplot titles and axes
            ax.set_title(f'σ = {sigma}, Fee Rate = {fee_rate}')
            ax.set_xlabel(self.formatted_name)
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
        
            if idx == 0:
                ax.legend()

        # Add overall title
        fig.suptitle(f'{self.formatted_name} Distribution for Sigma and Fee Rate Combinations', 
                     y=1.02, fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig
    
