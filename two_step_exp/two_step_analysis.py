"""Analysis class for two-step AMM calculations"""
import numpy as np
import polars as pl
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from .FeeRevenue import FeeRevenues
from .PoolValue import PoolValues
from .ArbitrageurRevenue import ArbitrageurRevenue

@dataclass
class ParamSet:
    """Parameter set for metric calculations"""
    sigma: float
    fee_rate: float
    fee_source: str

class TwoStepAnalysis:
    """Analysis class for two-step AMM calculations"""
    
    def __init__(self, ell_s: float = 1.0, ell_r: float = 1.0) -> None:
        """
        Initialize TwoStepAnalysis
        
        Args:
            ell_s: Stable token liquidity initial reserve
            ell_r: Risk token liquidity initial reserve
        """
        self.ell_s = ell_s
        self.ell_r = ell_r
        self.fee_revenue = FeeRevenues(ell_s, ell_r)
        self.pool_value = PoolValues(ell_s, ell_r)
        self.arbitrageur_revenue = ArbitrageurRevenue(ell_s, ell_r)

    def _calculate_expected_fee_revenue(self, sigma: float, f: float, fee_source: str) -> Dict[str, float]:
        """Calculate expected fee revenue by integrating over v1 and v2"""
        E_F1, E_F2 = self.fee_revenue.get_expected_fee_revenue(f, sigma, fee_source)
        return {
            'step1': E_F1,
            'step2': E_F1 + E_F2
        }
        
    def _calculate_expected_pool_value(self, sigma: float, f: float) -> Dict[str, float]:
        """Calculate expected pool value by integrating over v1 and v2"""
        E_PV1, E_PV2 = self.pool_value.get_expected_pool_value(f, sigma)
        return {
            'step1': E_PV1,
            'step2': E_PV2
        }
        
    def _calculate_expected_arbitrageur_revenue(self, sigma: float, f: float, fee_source: str) -> Dict[str, float]:
        """Calculate expected arbitrageur revenue by integrating over v1 and v2"""
        E_AR1, E_AR2 = self.arbitrageur_revenue.get_expected_arb_revenue(f, sigma, fee_source)
        return {
            'step1': E_AR1,
            'step2': E_AR1 + E_AR2
        }
        
    def calculate_metrics(self, sigma: float, f: float, fee_source: str) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for a single parameter set"""
        fee_revenue = self._calculate_expected_fee_revenue(sigma, f, fee_source)
        pool_value = self._calculate_expected_pool_value(sigma, f)
        arbitrageur_revenue = self._calculate_expected_arbitrageur_revenue(sigma, f, fee_source)
        
        # Calculate accounting profit for each step
        accounting_profit = {
            'step1': fee_revenue['step1'] + pool_value['step1'] - 2 * self.ell_s,
            'step2': fee_revenue['step2'] + pool_value['step2'] - 2 * self.ell_s,
        }
        
        return {
            'fee_revenue': fee_revenue,
            'pool_value': pool_value,
            'arbitrageur_revenue': arbitrageur_revenue,
            'accounting_profit': accounting_profit
        }

    @staticmethod
    def _process_param_set(args: Tuple[ParamSet, float, float]) -> List[Dict]:
        """Process a single parameter set (for parallel processing)"""
        param_set, ell_s, ell_r = args
        analyzer = TwoStepAnalysis(ell_s, ell_r)
        metrics = analyzer.calculate_metrics(param_set.sigma, param_set.fee_rate, param_set.fee_source)
        
        results = []
        # Add results for both steps
        for metric, values in metrics.items():
            for step in [1, 2]:
                results.append({
                    'metric': metric,
                    'step': step,
                    'fee_rate': param_set.fee_rate,
                    'sigma': param_set.sigma,
                    'fee_source': param_set.fee_source,
                    'value': values[f'step{step}']
                })
        return results
        
    def calculate_metrics_parallel(self, 
                             sigmas: Union[np.ndarray, List[float]], 
                             fee_rates: Union[np.ndarray, List[float]], 
                             output_path: str = 'output/two_step_metrics.parquet',
                             n_workers: Optional[int] = None) -> None:
        """
        Calculate metrics across parameter combinations using parallel processing
        
        Args:
            sigmas: Array of sigma values
            fee_rates: Array of fee rate values
            output_path: Path to save results
            n_workers: Number of worker processes (default: CPU count)
        """
        # Convert inputs to numpy arrays
        sigmas = np.array(sigmas)
        fee_rates = np.array(fee_rates)
        
        # Print setup info
        print("\n=== Two-Step AMM Analysis Setup ===")
        print(f"Sigma range: {min(sigmas):.1f} to {max(sigmas):.1f}, steps: {len(sigmas)}")
        print(f"Fee rate range: {min(fee_rates):.4f} to {max(fee_rates):.4f}, steps: {len(fee_rates)}")
        
        # Generate parameter combinations
        param_sets = []
        for fee_source in ['in', 'out']:
            for sigma in sigmas:
                for fee_rate in fee_rates:
                    param_sets.append(ParamSet(sigma, fee_rate, fee_source))

        # Prepare arguments for parallel processing
        args = [(param_set, self.ell_s, self.ell_r) for param_set in param_sets]
        n_workers = n_workers or cpu_count()
        
        print(f"\nTotal parameter combinations: {len(param_sets)}")
        print(f"Processing using {n_workers} CPU cores")
        
        # Process in parallel
        results = []
        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(param_sets), desc="Processing metrics", unit="combo") as pbar:
                for result in pool.imap_unordered(self._process_param_set, args, chunksize=10):
                    results.extend(result)
                    pbar.update(1)
        
        # Convert to DataFrame and save
        df = pl.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.write_csv(output_path.replace('.parquet', '.csv'))
        df.write_parquet(output_path, compression="snappy")
        
        print(f"\nSuccessfully saved metrics to {output_path}")
        print(f"Total rows: {len(df)}")
        
        # Print summary
        metrics_summary = df.group_by(['metric', 'step']).agg(
            pl.count('value').alias('rows')
        ).sort(['metric', 'step'])
        print("\nRows per metric and step:")
        for row in metrics_summary.iter_rows(named=True):
            print(f"  {row['metric']} (Step {row['step']}): {row['rows']:,}")
        
        
        