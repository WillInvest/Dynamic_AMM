import numpy as np
from scipy.stats import lognorm
from scipy.integrate import quad
import polars as pl
from typing import List, Optional, Dict, Union, Tuple
import os
from FeeRevenue import FeeRevenues
from PoolValue import PoolValues
from ArbitrageurRevenue import ArbitrageurRevenue



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
        """
        Calculate expected fee revenue by integrating over v1 and v2
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            fee_source: 'in' for incoming fees, 'out' for outgoing fees
            
        Returns:
            Dictionary containing expected fee revenue for step 1 and step 2
        """
        E_F1, E_F2 = self.fee_revenue.get_expected_fee_revenue(f, sigma, fee_source)
        return {
            'step1': E_F1,
            'step2': E_F1 + E_F2
        }
        
    def _calculate_expected_pool_value(self, sigma: float, f: float) -> Dict[str, float]:
        """
        Calculate expected pool value by integrating over v1 and v2
        """
        E_PV1, E_PV2 = self.pool_value.get_expected_pool_value(f, sigma)
        return {
            'step1': E_PV1,
            'step2': E_PV2
        }
        
    def _calculate_expected_arbitrageur_revenue(self, sigma: float, f: float, fee_source: str) -> Dict[str, float]:
        """
        Calculate expected arbitrageur revenue by integrating over v1 and v2
        """
        E_AR1, E_AR2 = self.arbitrageur_revenue.get_expected_arb_revenue(f, sigma, fee_source)
        return {
            'step1': E_AR1,
            'step2': E_AR1 + E_AR2
        }
        
    def calculate_metrics(self, sigma: float, f: float, fee_source: str) -> Dict[str, Tuple[float, float]]:
        """
        Calculate metrics for two-step AMM
        
        Args:
            sigma: Volatility parameter
            f: Fee rate
            
        Returns:
            Dictionary containing tuples of (step1, step2) values for each metric
        """
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
        
        
        
        