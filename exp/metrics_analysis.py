from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Union, TypedDict
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
from multiprocessing import Pool, cpu_count
import polars as pl
import os
from tqdm import tqdm
from datetime import datetime

class MetricResult(TypedDict):
    """Type definition for metric calculation results"""
    value: float
    vega: float

@dataclass
class ParamSet:
    """Parameter set for metric calculations"""
    sigma: float
    fee_rate: float
    is_ingoing: bool
    has_drift: bool

class MetricsAnalysis:
    """Analysis class for calculating various AMM metrics and their vegas"""
    
    def __init__(self, ell_s: float = 1.0) -> None:
        """
        Initialize MetricsAnalysis
        
        Args:
            ell_s: Liquidity scaling parameter
        """
        self.ell_s = ell_s
        self._available_metrics: Dict[str, Callable[[float, float, bool, bool], MetricResult]] = {
            'expected_fee': self._calculate_expected_fee,
            'pool_value': self._calculate_pool_value,
            'trader_pnl': self._calculate_trader_pnl,
            'impermanent_loss': self._calculate_impermanent_loss,
            'net_profit': self._calculate_net_profit,
            'accounting_profit': self._calculate_accounting_profit
        }
        print(f"Available metrics: {list(self._available_metrics.keys())}")

    @staticmethod
    def _lognormal_pdf(v: float, sigma: float, has_drift: bool) -> float:
        """Calculate lognormal PDF with or without drift"""
        if has_drift:
            return lognorm.pdf(v, s=sigma, scale=np.exp(0))
        return lognorm.pdf(v, s=sigma, scale=np.exp(-0.5 * sigma**2))
    
    @staticmethod
    def _sensitivity_term(v: float, sigma: float, has_drift: bool) -> float:
        """Calculate vega sensitivity term"""
        if has_drift:
            return (np.log(v)**2) / sigma**3 - 1 / sigma
        ln_v_term = np.log(v) + 0.5 * sigma**2
        return (ln_v_term**2) / sigma**3 - (ln_v_term + 1) / sigma
    
    @staticmethod
    def _weighted_sensitivity(v: float, sigma: float, has_drift: bool) -> float:
        """Calculate weighted sensitivity"""
        return MetricsAnalysis._lognormal_pdf(v, sigma, has_drift) * \
               MetricsAnalysis._sensitivity_term(v, sigma, has_drift)

    def _calculate_expected_fee(self, sigma: float, f: float, 
                              is_ingoing: bool, has_drift: bool) -> MetricResult:
        """
        Calculate expected fee and its vega
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            is_ingoing: Whether to calculate ingoing (True) or outgoing (False) fees
            has_drift: Whether to include drift term
            
        Returns:
            Dictionary containing value and vega
        """
        def _ingoing_price_effect(v: float, f: float, has_drift: bool) -> float:
            """Calculate price effect for ingoing fees"""
            if has_drift:
                return np.sqrt(1/(1-f)) * (np.sqrt(v) + 1/np.sqrt(v)) - (v + 1)/(1-f)
            return np.sqrt(v/(1-f)) - v/(1-f)
    
        def _outgoing_price_effect(v: float, f: float, has_drift: bool) -> float:
            """Calculate price effect for outgoing fees"""
            if has_drift:
                return -np.sqrt(1/(1-f)) * (np.sqrt(v) + 1/np.sqrt(v)) + (v + 1)/v
            return 1 - np.sqrt(v/(1-f))
        
        price_effect = _ingoing_price_effect if is_ingoing else _outgoing_price_effect
        
        # Calculate value
        def value_integrand(v: float) -> float:
            return (2 * f * self.ell_s * 
                   self._lognormal_pdf(v, sigma, has_drift) * 
                   price_effect(v, f, has_drift))
        value, _ = quad(lambda v: value_integrand(v), 1e-4, 1-f)
        
        # Calculate vega
        def vega_integrand(v: float) -> float:
            return (2 * f * self.ell_s * 
                   self._weighted_sensitivity(v, sigma, has_drift) * 
                   price_effect(v, f, has_drift))
        vega, _ = quad(lambda v: vega_integrand(v), 1e-4, 1-f)
        
        return {'value': value, 'vega': vega}

    def _calculate_pool_value(self, sigma: float, f: float, 
                              is_ingoing: bool, has_drift: bool) -> MetricResult:
        """
        Calculate pool value and its vega based on price movement integration.

        Args:
            sigma: Volatility parameter
            f: Fee rate
            is_ingoing: Whether to calculate ingoing (True) or outgoing (False) fees
            has_drift: Whether to include drift term (r_f = 0.5 * sigma^2 if True, 0 if False)

        Returns:
            Dictionary containing value and vega
        """
        def value_integrand_outside(v: float) -> float:
            """Integrand for the outside region [0, 1-f]"""
            if has_drift:
                # For positive drift: ℓs * (2-f)(1+v)/√((1-f)v) * φ(v)
                return (self.ell_s * (2-f) * (1+v) / np.sqrt((1-f)*v) * 
                       self._lognormal_pdf(v, sigma, has_drift))
            else:
                # For zero drift: 2ℓs * (√(v(1-f)) + √(v/(1-f))) * φ(v)
                return (2 * self.ell_s * 
                       (np.sqrt(v*(1-f)) + np.sqrt(v/(1-f))) * 
                       self._lognormal_pdf(v, sigma, has_drift))
        
        def value_integrand_inside(v: float) -> float:
            """Integrand for the inside region [1-f, 1/(1-f)]"""
            return (self.ell_s * (1+v) * 
                   self._lognormal_pdf(v, sigma, has_drift))
        
        def vega_integrand_outside(v: float) -> float:
            """Vega integrand for the outside region [0, 1-f]"""
            if has_drift:
                return (self.ell_s * (2-f) * (1+v) / np.sqrt((1-f)*v) * 
                       self._weighted_sensitivity(v, sigma, has_drift))
            else:
                return (2 * self.ell_s * 
                       (np.sqrt(v*(1-f)) + np.sqrt(v/(1-f))) * 
                       self._weighted_sensitivity(v, sigma, has_drift))
        
        def vega_integrand_inside(v: float) -> float:
            """Vega integrand for the inside region [1-f, 1/(1-f)]"""
            return (self.ell_s * (1+v) * 
                   self._weighted_sensitivity(v, sigma, has_drift))
        
        # Calculate the pool value using integration for both regions
        value_outside, _ = quad(value_integrand_outside, 1e-4, 1-f)
        value_inside, _ = quad(value_integrand_inside, 1-f, 1/(1-f))
        value = value_outside + value_inside

        # Calculate the vega using integration for both regions
        vega_outside, _ = quad(vega_integrand_outside, 1e-4, 1-f)
        vega_inside, _ = quad(vega_integrand_inside, 1-f, 1/(1-f))
        vega = vega_outside + vega_inside

        return {'value': value, 'vega': vega}

    def _calculate_trader_pnl(self, sigma: float, f: float, 
                             is_ingoing: bool, has_drift: bool) -> MetricResult:
        """
        Calculate trader PnL and its vega based on price movement integration
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            is_ingoing: Whether to calculate ingoing (True) or outgoing (False) fees
            has_drift: Whether to include drift term
            
        Returns:
            Dictionary containing value and vega
        """
        if is_ingoing:
            if has_drift:
                # Equation: ∫[0,1-f] ℓs·(1+v)·(√(1/v) + √(1/(1-f)))² φ(v) dv
                def value_integrand(v: float) -> float:
                    sqrt_term = np.sqrt(1/v) + np.sqrt(1/(1-f))
                    return self.ell_s * (1 + v) * sqrt_term**2 * self._lognormal_pdf(v, sigma, has_drift)
                
                def vega_integrand(v: float) -> float:
                    sqrt_term = np.sqrt(1/v) - np.sqrt(1/(1-f))
                    return self.ell_s * (1 + v) * sqrt_term**2 * self._weighted_sensitivity(v, sigma, has_drift)
            else:
                # Equation: ∫[0,1-f] 2ℓs·(1 - √(v/(1-f)))² φ(v) dv
                def value_integrand(v: float) -> float:
                    sqrt_term = 1 - np.sqrt(v/(1-f))
                    return 2 * self.ell_s * sqrt_term**2 * self._lognormal_pdf(v, sigma, has_drift)
                
                def vega_integrand(v: float) -> float:
                    sqrt_term = 1 - np.sqrt(v/(1-f))
                    return 2 * self.ell_s * sqrt_term**2 * self._weighted_sensitivity(v, sigma, has_drift)
        else:
            if has_drift:
                # Equation: ∫[0,1-f] ℓs·(1+v)·((1+v)/v - (2-f)/√(v(1-f))) φ(v) dv
                def value_integrand(v: float) -> float:
                    term1 = (1 + v) / v
                    term2 = (2 - f) / np.sqrt(v * (1-f))
                    return self.ell_s * (1 + v) * (term1 - term2) * self._lognormal_pdf(v, sigma, has_drift)
                
                def vega_integrand(v: float) -> float:
                    term1 = (1 + v) / v
                    term2 = (2 - f) / np.sqrt(v * (1-f))
                    return self.ell_s * (1 + v) * (term1 - term2) * self._weighted_sensitivity(v, sigma, has_drift)
            else:
                # Equation: ∫[0,1-f] 2ℓs·(1 - √(v/(1-f))·(2-f) + v) φ(v) dv
                def value_integrand(v: float) -> float:
                    return 2 * self.ell_s * (1 - np.sqrt(v/(1-f))*(2-f) + v) * self._lognormal_pdf(v, sigma, has_drift)
                
                def vega_integrand(v: float) -> float:
                    return 2 * self.ell_s * (1 - np.sqrt(v/(1-f))*(2-f) + v) * self._weighted_sensitivity(v, sigma, has_drift)

        # Calculate value and vega using numerical integration
        value, _ = quad(value_integrand, 1e-4, 1-f)
        vega, _ = quad(vega_integrand, 1e-4, 1-f)
        
        return {'value': value, 'vega': vega}

    def _calculate_impermanent_loss(self, sigma: float, f: float, 
                                  is_ingoing: bool, has_drift: bool) -> MetricResult:
        """
        Calculate impermanent loss and its vega based on price movement integration
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            is_ingoing: Whether to calculate ingoing (True) or outgoing (False) fees
            has_drift: Whether to include drift term
            
        Returns:
            Dictionary containing value and vega
        """
        # First calculate the expected pool value (PV₁)
        pool_value = self._calculate_pool_value(sigma, f, is_ingoing, has_drift)
        
        # Calculate E[1+v] based on drift case
        if has_drift:
            # For positive drift (r_f = σ²/2): E[1+v] = 1 + e^(σ²/2)
            expected_1_plus_v = 1 + np.exp(0.5 * sigma**2)
        else:
            # For zero drift (r_f = 0): E[1+v] = 2
            expected_1_plus_v = 2
        
        # Calculate IL value: E[IL] = E[PV₁] - ℓs·E[1+v]
        il_value = pool_value['value'] - self.ell_s * expected_1_plus_v
        
        # Calculate IL vega
        if has_drift:
            # For positive drift: ν_IL = ν_PV₁ - ℓs·σ·e^(σ²/2)
            il_vega = pool_value['vega'] - self.ell_s * sigma * np.exp(0.5 * sigma**2)
        else:
            # For zero drift: ν_IL = ν_PV₁
            il_vega = pool_value['vega']
        
        return {'value': il_value, 'vega': il_vega}

    def _calculate_net_profit(self, sigma: float, f: float, 
                            is_ingoing: bool, has_drift: bool) -> MetricResult:
        """
        Calculate net profit (fee revenue + impermanent loss) and its vega
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            is_ingoing: Whether to calculate ingoing (True) or outgoing (False) fees
            has_drift: Whether to include drift term
            
        Returns:
            Dictionary containing value and vega
        """
        # Calculate fee revenue
        fee_result = self._calculate_expected_fee(sigma, f, is_ingoing, has_drift)
        
        # Calculate impermanent loss
        il_result = self._calculate_impermanent_loss(sigma, f, is_ingoing, has_drift)
        
        # Net profit = fee revenue + impermanent loss
        net_profit_value = fee_result['value'] + il_result['value']
        
        # Vega of net profit = vega of fees + vega of IL
        net_profit_vega = fee_result['vega'] + il_result['vega']
        
        return {'value': net_profit_value, 'vega': net_profit_vega}

    def _calculate_accounting_profit(self, sigma: float, f: float, 
                               is_ingoing: bool, has_drift: bool) -> MetricResult:
        """
        Calculate accounting profit (fee revenue + pool value - initial investment) and its vega
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            is_ingoing: Whether to calculate ingoing (True) or outgoing (False) fees
            has_drift: Whether to include drift term
            
        Returns:
            Dictionary containing value and vega
        """
        # Calculate fee revenue
        fee_result = self._calculate_expected_fee(sigma, f, is_ingoing, has_drift)
        
        # Calculate pool value
        pool_value_result = self._calculate_pool_value(sigma, f, is_ingoing, has_drift)
        
        # Initial investment is 2ℓs
        initial_investment = 2 * self.ell_s
        
        # Accounting profit = fee revenue + pool value - initial investment
        accounting_profit_value = fee_result['value'] + pool_value_result['value'] - initial_investment
        
        # Vega of accounting profit = vega of fees + vega of pool value
        accounting_profit_vega = fee_result['vega'] + pool_value_result['vega']
        
        return {'value': accounting_profit_value, 'vega': accounting_profit_vega}

    def _generate_param_combinations(self, 
                                   sigmas: np.ndarray, 
                                   fee_rates: np.ndarray) -> List[ParamSet]:
        """Generate all parameter combinations for calculation"""
        params = []
        for sigma in sigmas:
            for fee_rate in fee_rates:
                for is_ingoing in [True, False]:
                    for has_drift in [True, False]:
                        params.append(ParamSet(sigma, fee_rate, is_ingoing, has_drift))
        return params

    def _calculate_metrics_for_params(self, 
                                    param_set: ParamSet, 
                                    metrics: List[str]) -> Dict[str, MetricResult]:
        """Calculate specified metrics for a parameter set"""
        return {
            metric: self._available_metrics[metric](
                param_set.sigma, param_set.fee_rate, 
                param_set.is_ingoing, param_set.has_drift
            )
            for metric in metrics
        }

    def _print_calculation_setup(self, metrics: List[str], existing_metrics: set,
                               sigmas: Union[np.ndarray, List[float]], 
                               fee_rates: Union[np.ndarray, List[float]]) -> None:
        """
        Print calculation setup information
        
        Args:
            metrics: List of metrics to calculate
            existing_metrics: Set of metrics that already exist
            sigmas: Array of sigma values
            fee_rates: Array of fee rate values
        """
        print("\n=== Metric Calculation Setup ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"New metrics to calculate: {', '.join(metrics)}")
        if existing_metrics:
            print(f"Existing metrics: {', '.join(existing_metrics)}")
        print(f"Sigma range: [{min(sigmas):.4f}, {max(sigmas):.4f}] ({len(sigmas)} values)")
        print(f"Fee rate range: [{min(fee_rates):.4f}, {max(fee_rates):.4f}] ({len(fee_rates)} values)")
        print(f"Fee Source: [ingoing, outgoing] (2 values)")
        print(f"Drift: [True, False] (2 values)")

    def get_metrics(self, 
                   sigmas: Union[np.ndarray, List[float]], 
                   fee_rates: Union[np.ndarray, List[float]], 
                   metrics: Optional[List[str]] = None, 
                   output_dir: str = 'output/metrics') -> None:
        """
        Calculate metrics and save each to a separate parquet file
        
        Args:
            sigmas: Array of sigma values
            fee_rates: Array of fee rate values
            metrics: List of metrics to calculate
            output_dir: Directory to save individual metric files
        """
        unknown_metrics = set(metrics) - set(self._available_metrics.keys())
        if unknown_metrics or metrics is None:
            raise ValueError(f"Unknown metrics: {unknown_metrics}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check which metrics already exist
        existing_metrics = set()
        for metric in metrics:
            metric_path = os.path.join(output_dir, f"{metric}.parquet")
            if os.path.exists(metric_path):
                existing_metrics.add(metric)
        
        # Filter out metrics that already exist
        metrics = [m for m in metrics if m not in existing_metrics]
        if not metrics:
            print("All requested metrics already exist!")
            return
        
        # Print calculation setup
        self._print_calculation_setup(metrics, existing_metrics, sigmas, fee_rates)

        # Generate parameters
        params = self._generate_param_combinations(np.array(sigmas), np.array(fee_rates))
        total_combinations = len(params)
        print(f"\nTotal parameter combinations: {total_combinations}")
        print(f"Processing using {cpu_count()} CPU cores")

        # Calculate metrics one at a time
        for metric in metrics:
            print(f"\nCalculating {metric}...")
            metric_results = []
            chunk_size = cpu_count() * 10
            
            with Pool(processes=cpu_count()) as pool:
                with tqdm(total=len(params), desc=f"Processing {metric}", unit="combo", ncols=80) as pbar:
                    for i in range(0, len(params), chunk_size):
                        chunk = params[i:i + chunk_size]
                        chunk_results = pool.starmap(self._calculate_metrics_for_params,
                                                  [(p, [metric]) for p in chunk])
                        
                        # Accumulate results
                        for param_set, result in zip(chunk, chunk_results):
                            fee_source = 'ingoing' if param_set.is_ingoing else 'outgoing'
                            values = result[metric]
                            metric_results.append({
                                'metric': metric,
                                'fee_rate': param_set.fee_rate,
                                'sigma': param_set.sigma,
                                'fee_source': fee_source,
                                'drift': param_set.has_drift,
                                'value': values['value'],
                                'vega': values['vega']
                            })
                        
                        pbar.update(len(chunk))
            
            # Save metric to separate file
            metric_df = pl.DataFrame(metric_results)
            metric_path = os.path.join(output_dir, f"{metric}.parquet")
            metric_df.write_parquet(metric_path, compression="snappy", statistics=True)
            print(f"Saved {metric} with {len(metric_results)} rows to {metric_path}")

    def _calculate_derived_metrics(self, row: Dict, fee: Tuple[float, float], 
                                 pool_value: Tuple[float, float]) -> List[Dict]:
        """
        Calculate derived metrics (IL, net profit, accounting profit) for a parameter set
        
        Args:
            row: Parameter combination (sigma, fee_rate, fee_source, drift)
            fee: Tuple of (fee_value, fee_vega)
            pool_value: Tuple of (pool_value, pool_value_vega)
            
        Returns:
            List of dictionaries containing derived metrics
        """
        sigma = row['sigma']
        has_drift = row['drift']
        
        # Calculate expected hold value
        if has_drift:
            expected_1_plus_v = 1 + np.exp(0.5 * sigma**2)
        else:
            expected_1_plus_v = 2
        hold_value = self.ell_s * expected_1_plus_v
        
        # Calculate all derived metrics
        derived_metrics = []
        
        # Impermanent Loss (pool value minus hold value)
        il_value = pool_value[0] - hold_value
        if has_drift:
            il_vega = pool_value[1] - self.ell_s * sigma * np.exp(0.5 * sigma**2)
        else:
            il_vega = pool_value[1]  # No vega component for hold value when no drift
        derived_metrics.append(self._create_metric_dict(
            'impermanent_loss', il_value, il_vega, row))
        
        # Net Profit
        net_profit_value = fee[0] + il_value
        net_profit_vega = fee[1] + il_vega
        derived_metrics.append(self._create_metric_dict(
            'net_profit', net_profit_value, net_profit_vega, row))
        
        # Accounting Profit
        accounting_profit_value = fee[0] + pool_value[0] - 2 * self.ell_s
        accounting_profit_vega = fee[1] + pool_value[1]
        derived_metrics.append(self._create_metric_dict(
            'accounting_profit', accounting_profit_value, accounting_profit_vega, row))
        
        return derived_metrics

    def _create_metric_dict(self, metric: str, value: float, vega: float, 
                           params: Dict) -> Dict:
        """Create a standardized metric dictionary"""
        return {
            'metric': metric,
            'sigma': params['sigma'],
            'fee_rate': params['fee_rate'],
            'fee_source': params['fee_source'],
            'drift': params['drift'],
            'value': value,
            'vega': vega
        }

    def combine_metrics(self, input_dir: str = 'output/metrics', 
                       output_path: str = 'output/combined_metrics.parquet') -> None:
        """
        Combine individual metric files into a single parquet file and derive additional metrics
        
        Args:
            input_dir: Directory containing individual metric files
            output_path: Path for combined output file
        """
        # Get base metric files
        base_metrics = ['expected_fee', 'pool_value', 'trader_pnl']
        metric_files = [f"{m}.parquet" for m in base_metrics]
        
        if not all(os.path.exists(os.path.join(input_dir, f)) for f in metric_files):
            raise ValueError(f"Missing required base metric files in {input_dir}")
        
        print("\n=== Combining Metrics ===")
        print(f"Reading base metrics: {', '.join(base_metrics)}")
        
        # Read base metrics and ensure consistent column order
        dfs = []
        expected_columns = ['metric', 'sigma', 'fee_rate', 'fee_source', 'drift', 'value', 'vega']
        for file in metric_files:
            df = pl.read_parquet(os.path.join(input_dir, file))
            # Ensure columns are in the correct order
            df = df.select(expected_columns)
            dfs.append(df)
        
        # Combine base metrics
        combined_df = pl.concat(dfs)
        
        # Create a DataFrame with unique parameter combinations
        params_df = combined_df.filter(pl.col('metric') == 'expected_fee').select(
            ['sigma', 'fee_rate', 'fee_source', 'drift']
        )
        
        # Get values for each base metric
        fee_df = combined_df.filter(pl.col('metric') == 'expected_fee')
        pool_value_df = combined_df.filter(pl.col('metric') == 'pool_value')
        
        # Calculate derived metrics
        derived_metrics = []
        print("Calculating derived metrics...")
        
        # Add progress bar
        for row in tqdm(params_df.iter_rows(named=True), 
                       total=len(params_df), 
                       desc="Deriving metrics", 
                       ncols=80):
            # Get base values
            fee = fee_df.filter(
                (pl.col('sigma') == row['sigma']) & 
                (pl.col('fee_rate') == row['fee_rate']) & 
                (pl.col('fee_source') == row['fee_source']) & 
                (pl.col('drift') == row['drift'])
            ).select('value', 'vega').row(0)
            
            pool_value = pool_value_df.filter(
                (pl.col('sigma') == row['sigma']) & 
                (pl.col('fee_rate') == row['fee_rate']) & 
                (pl.col('fee_source') == row['fee_source']) & 
                (pl.col('drift') == row['drift'])
            ).select('value', 'vega').row(0)
            
            derived_metrics.extend(
                self._calculate_derived_metrics(row, fee, pool_value)
            )
        
        # Add derived metrics to combined DataFrame
        derived_df = pl.DataFrame(derived_metrics)
        # Ensure derived metrics have the same column order
        derived_df = derived_df.select(expected_columns)
        final_df = pl.concat([combined_df, derived_df])
        
        # Save combined results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.write_parquet(output_path, compression="snappy", statistics=True)
        
        print(f"\nSuccessfully combined metrics into {output_path}")
        print(f"Total rows: {len(final_df)}")
        metrics_summary = final_df.group_by('metric').agg(
            pl.count('value').alias('rows')
        ).sort('metric')
        print("\nRows per metric:")
        for row in metrics_summary.iter_rows(named=True):
            print(f"  {row['metric']}: {row['rows']:,}")


if __name__ == "__main__":
    analyzer = MetricsAnalysis()
    # Only calculate base metrics
    base_metrics = [
        'expected_fee',
        'pool_value',
        'trader_pnl'
    ]
    analyzer.get_metrics(
        sigmas=np.round(np.arange(0.1, 8.1, 0.1), 1),
        fee_rates=np.linspace(0.0001, 0.9, 100),
        metrics=base_metrics,
        output_dir='/home/shiftpub/Dynamic_AMM/output/metrics'
    )
    
    # Combine and derive additional metrics
    analyzer.combine_metrics(
        input_dir='/home/shiftpub/Dynamic_AMM/output/metrics',
        output_path='/home/shiftpub/Dynamic_AMM/output/combined_metrics.parquet'
    )