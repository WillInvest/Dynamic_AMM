"""Calculation formulas for two-step AMM arbitrageur revenue"""
import numpy as np
from typing import Tuple
from scipy import integrate
from SwapReserve import SwapReserves

class ArbitrageurRevenue:
    """Class for calculating expected arbitrageur revenue"""
    
    def __init__(self, ell_s: float, ell_r: float):
        self.ell_s = ell_s
        self.ell_r = ell_r
        self.calc = SwapReserves(ell_s, ell_r)
        
    def phi(self, v: float, sigma: float) -> float:
        """PDF for geometric brownian motion with zero drift"""
        return 1/(v * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(v) + sigma**2/2)**2/(2*sigma**2))
        
    def get_arb_revenue(self, p: float, delta_s: float, delta_r: float, direction: str) -> float:
        """
        Calculate arbitrageur revenue for a given price and position changes
        
        Args:
            p: Current price
            delta_s: Change in stable token position
            delta_r: Change in risk token position
            direction: 'u' for upward price movement (sell r, buy s), 
                      'd' for downward price movement (sell s, buy r)
        """
        if direction == 'u':
            return delta_r * p - delta_s  # Sell r at p, buy s from AMM
        else:  # direction == 'd'
            return delta_s - delta_r * p  # Sell s to AMM, buy r at p
        
    def get_expected_arb_revenue(self, f: float, sigma: float, fee_source: str) -> Tuple[float, float]:
        """
        Calculate expected arbitrageur revenue for both steps
        
        Args:
            f: Fee rate
            sigma: Volatility
            
        Returns:
            Tuple of (E[V1], E[V2]) - expected arbitrageur revenue for step 1 and 2
        """
        # Step 1 expected arbitrageur revenue
        # Region: v1 > 1/(1-f)
        def integrand_up_1(v1: float) -> float:
            p = (self.ell_s / self.ell_r) * v1
            delta_s1, _ = self.calc.get_delta_s(f, fee_source, (v1, 'u'))
            delta_r1, _ = self.calc.get_delta_r(f, fee_source, (v1, 'u'))
            return self.get_arb_revenue(p, delta_s1, delta_r1, 'u') * self.phi(v1, sigma)
        
        # Region: v1 < 1-f
        def integrand_down_1(v1: float) -> float:
            p = (self.ell_s / self.ell_r) * v1
            delta_s1, _ = self.calc.get_delta_s(f, fee_source, (v1, 'd'))
            delta_r1, _ = self.calc.get_delta_r(f, fee_source, (v1, 'd'))
            return self.get_arb_revenue(p, delta_s1, delta_r1, 'd') * self.phi(v1, sigma)
            
        # Step 2 expected arbitrageur revenue - nine cases
        # Case 1: v1 > 1/(1-f), v2 > 1
        def integrand_uu(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'u'), (v2, 'u'))[1]
            delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'u'), (v2, 'u'))[1]
            return self.get_arb_revenue(p, delta_s2, delta_r2, 'u') * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 2: v1 > 1/(1-f), v2 < (1-f)^2
        def integrand_ud(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'u'), (v2, 'd'))[1]
            delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'u'), (v2, 'd'))[1]
            return self.get_arb_revenue(p, delta_s2, delta_r2, 'd') * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 3: 1-f < v1 < 1/(1-f), v2 > 1/(v1(1-f))
        def integrand_mu(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'm'), (v2, 'u'))[1]
            delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'm'), (v2, 'u'))[1]
            return self.get_arb_revenue(p, delta_s2, delta_r2, 'u') * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 4: 1-f < v1 < 1/(1-f), v2 < (1-f)/v1
        def integrand_md(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'm'), (v2, 'd'))[1]
            delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'm'), (v2, 'd'))[1]
            return self.get_arb_revenue(p, delta_s2, delta_r2, 'd') * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 5: v1 < 1-f, v2 > 1/(1-f)^2
        def integrand_du(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'd'), (v2, 'u'))[1]
            delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'd'), (v2, 'u'))[1]
            return self.get_arb_revenue(p, delta_s2, delta_r2, 'u') * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 6: v1 < 1-f, v2 < 1
        def integrand_dd(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            delta_s2 = self.calc.get_delta_s(f, fee_source, (v1, 'd'), (v2, 'd'))[1]
            delta_r2 = self.calc.get_delta_r(f, fee_source, (v1, 'd'), (v2, 'd'))[1]
            return self.get_arb_revenue(p, delta_s2, delta_r2, 'd') * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Step 1 integration
        E_V1_up = integrate.quad(integrand_up_1, 1/(1-f), np.inf)[0]
        E_V1_down = integrate.quad(integrand_down_1, 1e-4, 1-f)[0]
        E_V1 = E_V1_up + E_V1_down
        
        # Step 2 integration - all relevant cases
        # Upper region (v1 > 1/(1-f))
        E_V2_uu = integrate.dblquad(integrand_uu, 1/(1-f), np.inf, 
                                  lambda v1: 1, lambda v1: np.inf)[0]
        E_V2_ud = integrate.dblquad(integrand_ud, 1/(1-f), np.inf,
                                  lambda v1: 1e-4, lambda v1: (1-f)**2)[0]
        
        # Middle region (1-f < v1 < 1/(1-f))
        E_V2_mu = integrate.dblquad(integrand_mu, 1-f, 1/(1-f),
                                  lambda v1: 1/(v1*(1-f)), lambda v1: np.inf)[0]
        E_V2_md = integrate.dblquad(integrand_md, 1-f, 1/(1-f),
                                  lambda v1: 1e-4, lambda v1: (1-f)/v1)[0]
        
        # Lower region (v1 < 1-f)
        E_V2_du = integrate.dblquad(integrand_du, 1e-4, 1-f,
                                  lambda v1: 1/(1-f)**2, lambda v1: np.inf)[0]
        E_V2_dd = integrate.dblquad(integrand_dd, 1e-4, 1-f,
                                  lambda v1: 1e-4, lambda v1: 1)[0]
        
        E_V2 = (E_V2_uu + E_V2_ud + E_V2_mu + E_V2_md + E_V2_du + E_V2_dd)
        
        return E_V1, E_V2 