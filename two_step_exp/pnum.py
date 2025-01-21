"""Analysis class for calculating two-step AMM metrics across parameter combinations"""
import numpy as np
import polars as pl
from typing import List, Dict, Union, Optional, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from two_step_analysis import TwoStepAnalysis

@dataclass
class ParamSet:
    """Parameter set for metric calculations"""
    sigma: float
    fee_rate: float
    fee_source: str
    
class TwoStepNumericAnalysis:
    """Analysis class for calculating two-step AMM metrics"""
    
    def __init__(self, ell_s: float = 1.0, ell_r: float = 1.0):
        self.ell_s = ell_s
        self.ell_r = ell_r
        self.analyzer = TwoStepAnalysis(ell_s, ell_r)
        
    def _calculate_metrics_for_params(self, param_set: ParamSet) -> Dict[str, Tuple[float, float]]:
        """Calculate metrics for a given parameter set"""
        metrics = self.analyzer.calculate_metrics(param_set.sigma, param_set.fee_rate, param_set.fee_source)
        return {
            'fee_revenue': metrics['fee_revenue'],
            'pool_value': metrics['pool_value'],
            'arbitrageur_revenue': metrics['arbitrageur_revenue'],
            'accounting_profit': metrics['accounting_profit']
        }
        
    def _generate_param_combinations(self, 
                                  sigmas: np.ndarray, 
                                  fee_rates: np.ndarray) -> List[ParamSet]:
        """Generate all parameter combinations for calculation"""
        params = []
        for fee_source in ['in', 'out']:
            for sigma in sigmas:
                for fee_rate in fee_rates:
                    params.append(ParamSet(sigma, fee_rate, fee_source))
        return params
        
    def _print_calculation_setup(self, sigmas: np.ndarray, fee_rates: np.ndarray) -> None:
        """Print calculation setup information"""
        print("\n=== Two-Step AMM Analysis Setup ===")
        print(f"Sigma range: {min(sigmas):.1f} to {max(sigmas):.1f}, steps: {len(sigmas)}")
        print(f"Fee rate range: {min(fee_rates):.4f} to {max(fee_rates):.4f}, steps: {len(fee_rates)}")
        
    def calculate_metrics(self, 
                        sigmas: Union[np.ndarray, List[float]], 
                        fee_rates: Union[np.ndarray, List[float]], 
                        output_path: str = 'output/two_step_metrics.parquet') -> None:
        """
        Calculate metrics across parameter combinations and save to parquet file
        
        Args:
            sigmas: Array of sigma values
            fee_rates: Array of fee rate values
            output_path: Path to save results
        """
        # Convert inputs to numpy arrays if needed
        sigmas = np.array(sigmas)
        fee_rates = np.array(fee_rates)
        
        # Print calculation setup
        self._print_calculation_setup(sigmas, fee_rates)
        
        # Generate parameters
        params = self._generate_param_combinations(sigmas, fee_rates)
        total_combinations = len(params)
        print(f"\nTotal parameter combinations: {total_combinations}")
        print(f"Processing using {cpu_count()} CPU cores")
        
        # Calculate metrics using parallel processing
        results = []
        chunk_size = cpu_count() * 10
        
        with Pool(processes=cpu_count()) as pool:
            with tqdm(total=len(params), desc="Processing metrics", unit="combo", ncols=100) as pbar:
                for i in range(0, len(params), chunk_size):
                    chunk = params[i:i + chunk_size]
                    chunk_results = pool.map(self._calculate_metrics_for_params, chunk)
                    
                    # Accumulate results
                    for param_set, metrics in zip(chunk, chunk_results):
                        # Add step 1 results
                        results.extend([
                            {
                                'metric': metric,
                                'step': 1,
                                'fee_rate': param_set.fee_rate,
                                'sigma': param_set.sigma,
                                'fee_source': param_set.fee_source,
                                'value': values['step1']  # Step 1 value
                            }
                            for metric, values in metrics.items()
                        ])
                        
                        # Add step 2 results
                        results.extend([
                            {
                                'metric': metric,
                                'step': 2,
                                'fee_rate': param_set.fee_rate,
                                'sigma': param_set.sigma,
                                'fee_source': param_set.fee_source,
                                'value': values['step2']  # Step 2 value
                            }
                            for metric, values in metrics.items()
                        ])
                    
                    pbar.update(len(chunk))
        
        # Convert results to DataFrame and save
        df = pl.DataFrame(results)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        df.write_parquet(output_path, compression="snappy", statistics=True)
        
        print(f"\nSuccessfully saved metrics to {output_path}")
        print(f"Total rows: {len(df)}")
        metrics_summary = df.group_by(['metric', 'step']).agg(
            pl.count('value').alias('rows')
        ).sort(['metric', 'step'])
        print("\nRows per metric and step:")
        for row in metrics_summary.iter_rows(named=True):
            print(f"  {row['metric']} (Step {row['step']}): {row['rows']:,}")

if __name__ == "__main__":

    # sigmas = np.concatenate([np.round(np.arange(0.01, 0.1, 0.01), 2),
    #                          np.round(np.arange(0.1, 1, 0.1), 1)])
    sigmas = np.round(np.arange(0.1, 1, 0.1), 1)
    fee_rates = np.linspace(0.0001, 0.8, 100)
    
    analyzer = TwoStepNumericAnalysis()
    analyzer.calculate_metrics(
        sigmas=sigmas,
        fee_rates=fee_rates,
        output_path='output/two_step_metrics.parquet'
    ) 