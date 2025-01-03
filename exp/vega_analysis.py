import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad, cumulative_trapezoid
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os 
import time


class VegaAnalysis:
    def __init__(self, fee_rate=0.03, ell_s=1, save=True, show=False):
        self.fee_rate = fee_rate
        self.ell_s = ell_s
        self.save = save
        self.show = show
        
    @staticmethod
    def lognormal_pdf(v, sigma):
        """Common lognormal PDF"""
        return lognorm.pdf(v, s=sigma, scale=np.exp(0))
    
    @staticmethod
    def sensitivity_term(v, sigma):
        """Common sensitivity term"""
        return (np.log(v)**2) / sigma**3 - 1 / sigma
    
    @staticmethod
    def weighted_sensitivity(v, sigma):
        """Common weighted sensitivity"""
        return VegaAnalysis.lognormal_pdf(v, sigma) * VegaAnalysis.sensitivity_term(v, sigma)
    
    def ingoing_price_effect(self, v):
        """Price effect for ingoing fees"""
        f = self.fee_rate
        return np.sqrt(1/(1-f)) * (np.sqrt(v) + 1/np.sqrt(v)) - (v + 1)/(1-f)
    
    def outgoing_price_effect(self, v):
        """Price effect for outgoing fees"""
        f = self.fee_rate
        return -np.sqrt(1/(1-f)) * (np.sqrt(v) + 1/np.sqrt(v)) + (v + 1)/v
    
    def calculate_vega(self, sigma, is_ingoing=True):
        """Calculate vega for given sigma"""
        f = self.fee_rate
        price_effect = self.ingoing_price_effect if is_ingoing else self.outgoing_price_effect
        
        def integrand(v):
            return (self.weighted_sensitivity(v, sigma) * 
                   price_effect(v) * 
                   f * 
                   self.ell_s)
        
        result, _ = quad(lambda v: integrand(v), 1e-4, 1-f)
        return result
    
    def calculate_component_effect(self, v_points, sigma, component='full', is_ingoing=True):
        """Calculate different component effects for analysis"""
        if component == 'lognormal':
            return self.lognormal_pdf(v_points, sigma)
        elif component == 'sensitivity':
            return self.sensitivity_term(v_points, sigma)
        elif component == 'weighted_sensitivity':
            return self.weighted_sensitivity(v_points, sigma)
        elif component == 'full':
            price_effect = self.ingoing_price_effect if is_ingoing else self.outgoing_price_effect
            return (self.weighted_sensitivity(v_points, sigma) * 
                   price_effect(v_points) * 
                   self.fee_rate * 
                   self.ell_s)
    
    def plot_component_analysis(self, max_sigma=4.0, is_ingoing=True):
        """Plot component analysis"""
        # Create output directory
        fee_type = "ingoing" if is_ingoing else "outgoing"
        os.makedirs(f'{fee_type}_analysis', exist_ok=True)
        print(f"Plotting {fee_type} component analysis...")
        sigmas = np.linspace(0.1, max_sigma, 100)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        components = ['lognormal', 'sensitivity', 'weighted_sensitivity', 'full']
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Log-normal PDF\nIntegral', 
                 'Sensitivity Term\nIntegral',
                 'Weighted Sensitivity Terms\nIntegral',
                 'Combined Term\nIntegral']
        
        for ax, component, title in zip(axes, components, titles):
            values = [self.calculate_vega(sigma, is_ingoing) 
                     if component == 'full' 
                     else quad(lambda v: self.calculate_component_effect(v, sigma, component), 
                             1e-4, 1-self.fee_rate)[0] 
                     for sigma in sigmas]
            ax.plot(sigmas, values, linewidth=2)
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        fee_type = "ingoing" if is_ingoing else "outgoing"
        plt.suptitle(f'{fee_type} Component Analysis (Fee Rate: {self.fee_rate*10000:.0f} bps)', 
                    fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        filename = f'{fee_type}_analysis/component_analysis_max_sigma_{max_sigma}.png'
        self.save_and_show(filename)

        
    def plot_price_effect_cumulative(self, is_ingoing=True):
        """Plot cumulative price effect for either ingoing or outgoing fees"""
        # Create output directory
        fee_type = "ingoing" if is_ingoing else "outgoing"
        os.makedirs(f'{fee_type}_analysis', exist_ok=True)
        print(f"Plotting {fee_type} price effect cumulative...")
        
        v_points = np.linspace(1e-4, 1-self.fee_rate, 1000)
        
        # Calculate price effect and cumulative integral
        price_effect = self.ingoing_price_effect if is_ingoing else self.outgoing_price_effect
        effect_values = price_effect(v_points)
        cumulative = cumulative_trapezoid(effect_values, v_points, initial=0)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(v_points, cumulative, linewidth=2)
        
        # Set labels and title
        fee_type_title = "Ingoing" if is_ingoing else "Outgoing"
        plt.title(f'Cumulative {fee_type_title} Price Effect (Fee Rate: {self.fee_rate*10000:.0f} bps)')
        plt.xlabel('Integration Domain (dv)')
        plt.ylabel('Cumulative Value')
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # Save plot
        filename = f'{fee_type}_analysis/price_effect_cumulative.png'
        self.save_and_show(filename)

        
    def plot_cumulative_effect(self, sigmas, component='price', is_ingoing=True):
        """Plot cumulative effect analysis"""
        # Create output directory
        fee_type = "ingoing" if is_ingoing else "outgoing"
        os.makedirs(f'{fee_type}_analysis', exist_ok=True)
        print(f"Plotting {fee_type} cumulative effect analysis...")
        v_points = np.linspace(1e-4, 1-self.fee_rate, int(1e6))
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for idx, sigma in enumerate(sigmas):
            ax = axes[idx]
            values = self.calculate_component_effect(v_points, sigma, component, is_ingoing)
            cumulative = cumulative_trapezoid(values, v_points, initial=0)
            
            ax.plot(v_points, cumulative, linewidth=2, color='blue')
            ax.set_title(f'σ = {sigma:.1f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        fee_type = "ingoing" if is_ingoing else "outgoing"
        component_type = component.replace('_', ' ').title()
        if component in ['lognormal', 'sensitivity', 'weighted_sensitivity']:
            plt.suptitle(f'{component_type} Analysis | Fee Rate: {self.fee_rate*10000:.0f} bps', 
                        fontsize=16)
        else:
            plt.suptitle(f'{fee_type} {component_type} Effect Analysis | Fee Rate: {self.fee_rate*10000:.0f} bps', 
                        fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        filename = f'{fee_type}_analysis/cumulative_{component}.png'
        self.save_and_show(filename)

        
    def calculate_vegas_parallel(self, sigmas, fee_rates, is_ingoing=True):
        """Calculate vegas for all combinations of sigma and fee rates in parallel"""
        params = [(sigma, f, is_ingoing) for f in fee_rates for sigma in sigmas]
        
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(self.calculate_vega_single, params)
        
        return np.array(results).reshape(len(fee_rates), len(sigmas))
    
    def calculate_vega_single(self, sigma, fee_rate, is_ingoing=True):
        """Calculate vega for a single combination of sigma and fee rate"""
        old_fee_rate = self.fee_rate
        self.fee_rate = fee_rate
        try:
            result = self.calculate_vega(sigma, is_ingoing)
            self.fee_rate = old_fee_rate
            return result
        except Exception as e:
            print(f"Error at sigma={sigma}, f={fee_rate}: {e}")
            self.fee_rate = old_fee_rate
            return np.nan

    def plot_vega_analysis(self, max_sigma=5.0, is_ingoing=True):
        """Plot vega analysis for different fee rates and volatilities"""
        # Create output directory
        fee_type = "ingoing" if is_ingoing else "outgoing"
        os.makedirs(f'vega_plots_{fee_type}', exist_ok=True)
        print(f"Ploting {fee_type} Vegas...")

        # Parameters
        # fee_rates = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03,
        #             0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        #             0.55, 0.6, 0.7, 0.8, 0.9]
        fee_rates = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        
        sigmas = np.arange(0.1, max_sigma+0.1, 0.1)
        if len(sigmas) < 50:
            sigmas = np.linspace(0.1, max_sigma, 50)        
        
        vegas_matrix = self.calculate_vegas_parallel(sigmas, fee_rates, is_ingoing)
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for idx, (fee, vegas) in enumerate(zip(fee_rates, vegas_matrix)):
            ax = axes[idx]
            ax.plot(sigmas, vegas, linewidth=2)
            ax.set_title(f'Fee Rate: {fee*10000:.0f} bps', fontsize=12)
            ax.set_xlabel('Volatility (σ)', fontsize=10)
            ax.set_ylabel('Vega', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.grid(True, which='minor', alpha=0.1)
            ax.grid(True, which='major', alpha=0.3)
            ax.set_xlim(sigmas[0], sigmas[-1])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        fig.suptitle(f'{fee_type} vega | Volatility Range = [0.1, {max_sigma}]', fontsize=16)
        
        filename = f'vega_plots_{fee_type}/{fee_type}_vega_subplots_max_sigma_{max_sigma}.png'
        self.save_and_show(filename)

        
    def plot_cumulative_comparison_analysis(self, is_ingoing=True):
        """Plot sensitivity vega analysis with weighted sensitivity, aggregate term, and final values"""
        # Create output directory
        fee_type = "ingoing" if is_ingoing else "outgoing"
        os.makedirs(f'{fee_type}_analysis', exist_ok=True)
        print(f"Plotting {fee_type} cumulative comparison analysis...")
        
        v_points = np.linspace(1e-4, 1-self.fee_rate, int(1e6))
        sigmas = [0.2, 0.4, 0.6]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        vegas = []
        for sigma, color in zip(sigmas, colors):
            # First plot: Weighted sensitivity
            wsens_values = self.weighted_sensitivity(v_points, sigma)
            cumulative1 = cumulative_trapezoid(wsens_values, v_points, initial=0)
            ax1.plot(v_points, cumulative1, linewidth=2, label=f'σ = {sigma}', color=color)
            
            # Second plot: Aggregate term
            price_effect = self.ingoing_price_effect if is_ingoing else self.outgoing_price_effect
            agg_values = (self.weighted_sensitivity(v_points, sigma) * 
                         price_effect(v_points) * 
                         self.fee_rate)
            cumulative2 = cumulative_trapezoid(agg_values, v_points, initial=0)
            ax2.plot(v_points, cumulative2, linewidth=2, label=f'σ = {sigma}', color=color)
            
            vegas.append(cumulative2[-1])
        
        # Third plot: Bar plot of final values
        ax3.bar(range(len(sigmas)), vegas, alpha=0.3, width=0.6, color=colors)
        ax3.set_xticks(range(len(sigmas)))
        ax3.set_xticklabels([f'σ={s}' for s in sigmas])
        
        # Set titles and labels
        ax1.set_title('Weighted Sensitivity Term')
        ax2.set_title('Aggregate Term')
        ax3.set_title('Final Vega Values')
        
        ax1.set_xlabel('Integration Domain (dv)')
        ax2.set_xlabel('Integration Domain (dv)')
        ax3.set_xlabel('Volatility')
        
        ax1.set_ylabel('Cumulative Value')
        ax2.set_ylabel('Cumulative Value')
        ax3.set_ylabel('Vega')
        
        # Customize axes
        ax3.set_ylim(bottom=min(vegas)*0.95, top=max(vegas)*1.05)
        
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        
        # Add overall title
        fee_type_title = "ingoing" if is_ingoing else "outgoing"
        plt.suptitle(f'{fee_type_title} Cumulative Comparison Analysis (Fee Rate: {self.fee_rate*10000:.0f} bps)', 
                    fontsize=14, y=1.05)
        
        # Save the plot
        filename = f'{fee_type}_analysis/cumulative_comparison_analysis.png'
        self.save_and_show(filename)

        
    def plot_vega_vs_fee(self, max_sigma=5, max_fee=0.99, is_ingoing=True):
        """Plot vega values against fee rates for different volatilities"""
        # Create output directory
        fee_type = "ingoing" if is_ingoing else "outgoing"
        os.makedirs(f'{fee_type}_analysis', exist_ok=True)
        print(f"Plotting {fee_type} vega vs fee rate...")
        
        # Generate parameter ranges
        volatilities = np.linspace(0.2, max_sigma, 9)
        fee_rates = np.linspace(0.0005, max_fee, 50)
        
        # Generate all parameter combinations
        params = [(sigma, f, is_ingoing) for sigma in volatilities for f in fee_rates]
        
        # Calculate vegas in parallel
        with Pool(processes=cpu_count()) as pool:
            vegas = pool.starmap(self.calculate_vega_single, params)
        
        # Reshape results
        vegas_matrix = np.array(vegas).reshape(len(volatilities), len(fee_rates))
        
        # Plot
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, (sigma, vega_row) in enumerate(zip(volatilities, vegas_matrix)):
            ax = axes[idx]
            ax.plot(fee_rates * 10000, vega_row, linewidth=2)
            ax.set_title(f'σ = {sigma:.1f}', fontsize=12)
            
            if idx % 5 == 0:
                ax.set_ylabel('Vega', fontsize=10)
            if idx >= 20:
                ax.set_xlabel('Fee Rate (bps)', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        fee_type_title = "ingoing" if is_ingoing else "outgoing"
        plt.suptitle(f'{fee_type_title} Vega vs Fee Rate (σ range: [0.2, {max_sigma}])', 
                    fontsize=16)
        
        # Save the plot
        filename = f'{fee_type}_analysis/vega_vs_fee_max_sigma_{max_sigma}.png'
        self.save_and_show(filename)
    
    def save_and_show(self, filename):
        if self.save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        if self.show:
            plt.show()
        plt.close()

if __name__ == '__main__':
    analyzer = VegaAnalysis()

    # Generate vega plots for different volatility ranges
    for max_sigma in [1.0, 5.0, 10.0]:
        for is_ingoing in [True, False]:
            analyzer.plot_vega_analysis(max_sigma=max_sigma, is_ingoing=is_ingoing)

    # Original analysis plots
    for is_ingoing in [True, False]:
        analyzer.plot_component_analysis(max_sigma=4.0, is_ingoing=is_ingoing)
        analyzer.plot_cumulative_comparison_analysis(is_ingoing=is_ingoing)
        analyzer.plot_vega_vs_fee(is_ingoing=is_ingoing)
        analyzer.plot_price_effect_cumulative(is_ingoing=is_ingoing)



    sigmas = np.linspace(0.1, 4.0, 9)
    components = ['lognormal', 'sensitivity', 'weighted_sensitivity', 'full']
    for component in components:
        for is_ingoing in [True, False]:
            analyzer.plot_cumulative_effect(sigmas, component, is_ingoing)
