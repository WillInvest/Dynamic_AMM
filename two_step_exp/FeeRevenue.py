"""Calculation formulas for two-step AMM fee revenues"""
import numpy as np
from typing import Tuple
from scipy import integrate
from SwapReserve import SwapReserves

class FeeRevenues:
    """Class for calculating expected fee revenue"""
    
    def __init__(self, ell_s: float, ell_r: float):
        self.ell_s = ell_s
        self.ell_r = ell_r
        self.calc = SwapReserves(ell_s, ell_r)
        
    def phi(self, v: float, sigma: float) -> float:
        """PDF for geometric brownian motion with zero drift"""
        return 1/(v * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(v) + sigma**2/2)**2/(2*sigma**2))
    
    def get_expected_fee_revenue(self, f: float, sigma: float, fee_source: str) -> Tuple[float, float]:
        """
        Calculate expected fee revenue for both steps
        
        Args:
            f: Fee rate
            sigma: Volatility
            fee_source: 'in' for incoming fees, 'out' for outgoing fees
            
        Returns:
            Tuple of (E[F1], E[F2]) - expected fee revenue for step 1 and 2
        """
        # make sure the fee source is either 'in' or 'out'
        if fee_source not in ['in', 'out']:
            raise ValueError("fee_source must be either 'in' or 'out'")
        
        # Step 1 expected fee revenue
        # Region: v1 > 1/(1-f)
        def integrand_up_1(v1: float) -> float:
            p = (self.ell_s / self.ell_r) * v1
            if fee_source == 'in':
                delta_s1, _ = self.calc.get_delta_s(f, fee_source, (v1, 'u'))
                return delta_s1 * f * self.phi(v1, sigma)
            else:  # fee_source == 'out'
                delta_r1, _ = self.calc.get_delta_r(f, fee_source, (v1, 'u'))
                return delta_r1 * p * f * self.phi(v1, sigma)
        
        # Region: 1-f < v1 < 1/(1-f)
        def integrand_mid_1(v1: float) -> float:
            return 0  # No arbitrage in middle region
        
        # Region: v1 < 1-f
        def integrand_down_1(v1: float) -> float:   
            p = (self.ell_s / self.ell_r) * v1
            if fee_source == 'in':
                delta_r1, _ = self.calc.get_delta_r(f, fee_source, (v1, 'd'))
                return delta_r1 * p * f * self.phi(v1, sigma)
            else:  # fee_source == 'out'
                delta_s1, _ = self.calc.get_delta_s(f, fee_source, (v1, 'd'))
                return delta_s1 * f * self.phi(v1, sigma)
        
        # Step 2 expected fee revenue - nine cases
        # Case 1: v1 > 1/(1-f), v2 > 1
        def integrand_uu(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            if fee_source == 'in':
                delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'u'), (v2, 'u'))[1]
                return delta_s2 * f * self.phi(v1, sigma) * self.phi(v2, sigma)
            else:
                delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'u'), (v2, 'u'))[1]
                return delta_r2 * p * f * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Case 2: v1 > 1/(1-f), (1-f)^2 < v2 < 1
        def integrand_um(v1: float, v2: float) -> float:
            return 0  # No arbitrage
        
        # Case 3: v1 > 1/(1-f), v2 < (1-f)^2
        def integrand_ud(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            if fee_source == 'in':
                delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'u'), (v2, 'd'))[1]
                return delta_r2 * p * f * self.phi(v1, sigma) * self.phi(v2, sigma)
            else:
                delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'u'), (v2, 'd'))[1]
                return delta_s2 * f * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Case 4: 1-f < v1 < 1/(1-f), v2 > 1/(v1*(1-f))
        def integrand_mu(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            if fee_source == 'in':
                delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'm'), (v2, 'u'))[1]
                return delta_s2 * f * self.phi(v1, sigma) * self.phi(v2, sigma)
            else:
                delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'm'), (v2, 'u'))[1]
                return delta_r2 * p * f * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Case 5: 1-f < v1 < 1/(1-f), (1-f)/v1 < v2 < 1/(v1*(1-f))
        def integrand_mm(v1: float, v2: float) -> float:
            return 0  # No arbitrage
        
        # Case 6: 1-f < v1 < 1/(1-f), v2 < (1-f)/v1
        def integrand_md(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            if fee_source == 'in':
                delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'm'), (v2, 'd'))[1]
                return delta_r2 * p * f * self.phi(v1, sigma) * self.phi(v2, sigma)
            else:
                delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'm'), (v2, 'd'))[1]
                return delta_s2 * f * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Case 7: v1 < 1-f, v2 > 1/(1-f)^2
        def integrand_du(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            if fee_source == 'in':
                delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'd'), (v2, 'u'))[1]
                return delta_s2 * f * self.phi(v1, sigma) * self.phi(v2, sigma)
            else:
                delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'd'), (v2, 'u'))[1]
                return delta_r2 * p * f * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Case 8: v1 < 1-f, 1 < v2 < 1/(1-f)^2
        def integrand_dm(v1: float, v2: float) -> float:
            return 0  # No arbitrage
        
        # Case 9: v1 < 1-f, v2 < 1
        def integrand_dd(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            if fee_source == 'in':
                delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'd'), (v2, 'd'))[1]
                return delta_r2 * p * f * self.phi(v1, sigma) * self.phi(v2, sigma)
            else:
                delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'd'), (v2, 'd'))[1]
                return delta_s2 * f * self.phi(v1, sigma) * self.phi(v2, sigma)
                
        # Step 1 integration
        E_F1_up = integrate.quad(integrand_up_1, 1/(1-f), np.inf)[0]
        E_F1_mid = integrate.quad(integrand_mid_1, 1-f, 1/(1-f))[0]
        E_F1_down = integrate.quad(integrand_down_1, 1e-4, 1-f)[0]
        E_F1 = E_F1_up + E_F1_mid + E_F1_down
        
        # Step 2 integration - all nine cases
        # Upper region (v1 > 1/(1-f))
        E_F2_uu = integrate.dblquad(integrand_uu, 1/(1-f), np.inf, lambda v1: 1, lambda v1: np.inf)[0]
        E_F2_um = integrate.dblquad(integrand_um, 1/(1-f), np.inf, lambda v1: (1-f)**2, lambda v1: 1)[0]
        E_F2_ud = integrate.dblquad(integrand_ud, 1/(1-f), np.inf, lambda v1: 1e-4, lambda v1: (1-f)**2)[0]
        
        # Middle region (1-f < v1 < 1/(1-f))
        E_F2_mu = integrate.dblquad(integrand_mu, 1-f, 1/(1-f), 
                                  lambda v1: 1/(v1*(1-f)), lambda v1: np.inf)[0]
        E_F2_mm = integrate.dblquad(integrand_mm, 1-f, 1/(1-f),
                                  lambda v1: (1-f)/v1, lambda v1: 1/(v1*(1-f)))[0]
        E_F2_md = integrate.dblquad(integrand_md, 1-f, 1/(1-f),
                                  lambda v1: 1e-4, lambda v1: (1-f)/v1)[0]
        
        # Lower region (v1 < 1-f)
        E_F2_du = integrate.dblquad(integrand_du, 0, 1-f,
                                  lambda v1: 1/(1-f)**2, lambda v1: np.inf)[0]
        E_F2_dm = integrate.dblquad(integrand_dm, 0, 1-f,
                                  lambda v1: 1, lambda v1: 1/(1-f)**2)[0]
        E_F2_dd = integrate.dblquad(integrand_dd, 0, 1-f,
                                  lambda v1: 1e-4, lambda v1: 1)[0]
        
        E_F2 = E_F2_uu + E_F2_um + E_F2_ud + E_F2_mu + E_F2_mm + E_F2_md + E_F2_du + E_F2_dm + E_F2_dd
        
        return E_F1, E_F2
    