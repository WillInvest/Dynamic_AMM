"""Calculation formulas for two-step AMM pool values"""
import numpy as np
from typing import Tuple
from scipy import integrate
from SwapReserve import SwapReserves

class PoolValues:
    """Class for calculating expected pool values"""
    
    def __init__(self, ell_s: float, ell_r: float):
        self.ell_s = ell_s
        self.ell_r = ell_r
        self.calc = SwapReserves(ell_s, ell_r)
        
    def phi(self, v: float, sigma: float) -> float:
        """PDF for geometric brownian motion with zero drift"""
        return 1/(v * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(v) + sigma**2/2)**2/(2*sigma**2))
        
    def get_pool_value(self, p: float, ls: float, lr: float) -> float:
        """Calculate pool value at given price and reserves"""
        return lr * p + ls
        
    def get_expected_pool_value(self, f: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate expected pool value for both steps
        
        Args:
            f: Fee rate
            sigma: Volatility
            
        Returns:
            Tuple of (E[PV1], E[PV2]) - expected pool values for step 1 and 2
        """
        # Step 1 expected pool value
        # Region: v1 > 1/(1-f)
        def integrand_up_1(v1: float) -> float:
            p = (self.ell_s / self.ell_r) * v1
            ls1, _ = self.calc.get_ls(f, (v1, 'u'))
            lr1, _ = self.calc.get_lr(f, (v1, 'u'))
            return self.get_pool_value(p, ls1, lr1) * self.phi(v1, sigma)
        
        # Region: 1-f < v1 < 1/(1-f)
        def integrand_mid_1(v1: float) -> float:
            p = (self.ell_s / self.ell_r) * v1
            ls1, _ = self.calc.get_ls(f, (v1, 'm'))
            lr1, _ = self.calc.get_lr(f, (v1, 'm'))
            return self.get_pool_value(p, ls1, lr1) * self.phi(v1, sigma)
        
        # Region: v1 < 1-f
        def integrand_down_1(v1: float) -> float:
            p = (self.ell_s / self.ell_r) * v1
            ls1, _ = self.calc.get_ls(f, (v1, 'd'))
            lr1, _ = self.calc.get_lr(f, (v1, 'd'))
            return self.get_pool_value(p, ls1, lr1) * self.phi(v1, sigma)
            
        # Step 2 expected pool value - nine cases
        # Case 1: v1 > 1/(1-f), v2 > 1
        def integrand_uu(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'u'), (v2, 'u'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'u'), (v2, 'u'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 2: v1 > 1/(1-f), (1-f)^2 < v2 < 1
        def integrand_um(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'u'), (v2, 'm'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'u'), (v2, 'm'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 3: v1 > 1/(1-f), v2 < (1-f)^2
        def integrand_ud(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'u'), (v2, 'd'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'u'), (v2, 'd'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 4: 1-f < v1 < 1/(1-f), v2 > 1/(v1(1-f))
        def integrand_mu(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'm'), (v2, 'u'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'm'), (v2, 'u'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 5: 1-f < v1 < 1/(1-f), (1-f)/v1 < v2 < 1/(v1(1-f))
        def integrand_mm(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'm'), (v2, 'm'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'm'), (v2, 'm'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 6: 1-f < v1 < 1/(1-f), v2 < (1-f)/v1
        def integrand_md(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'm'), (v2, 'd'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'm'), (v2, 'd'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 7: v1 < 1-f, v2 > 1/(1-f)^2
        def integrand_du(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'd'), (v2, 'u'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'd'), (v2, 'u'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 8: v1 < 1-f, 1 < v2 < 1/(1-f)^2
        def integrand_dm(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'd'), (v2, 'm'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'd'), (v2, 'm'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
            
        # Case 9: v1 < 1-f, v2 < 1
        def integrand_dd(v1: float, v2: float) -> float:
            p = (self.ell_s / self.ell_r) * v1 * v2
            ls2 = self.calc.get_ls(f, (v1, 'd'), (v2, 'd'))[1]
            lr2 = self.calc.get_lr(f, (v1, 'd'), (v2, 'd'))[1]
            return self.get_pool_value(p, ls2, lr2) * self.phi(v1, sigma) * self.phi(v2, sigma)
        
        # Step 1 integration
        E_PV1_up = integrate.quad(integrand_up_1, 1/(1-f), np.inf)[0]
        E_PV1_mid = integrate.quad(integrand_mid_1, 1-f, 1/(1-f))[0]
        E_PV1_down = integrate.quad(integrand_down_1, 1e-4, 1-f)[0]
        E_PV1 = E_PV1_up + E_PV1_mid + E_PV1_down
        
        # Step 2 integration - all nine cases
        # Upper region (v1 > 1/(1-f))
        E_PV2_uu = integrate.dblquad(integrand_uu, 1/(1-f), np.inf, lambda v1: 1, lambda v1: np.inf)[0]
        E_PV2_um = integrate.dblquad(integrand_um, 1/(1-f), np.inf, lambda v1: (1-f)**2, lambda v1: 1)[0]
        E_PV2_ud = integrate.dblquad(integrand_ud, 1/(1-f), np.inf, lambda v1: 1e-4, lambda v1: (1-f)**2)[0]
        
        # Middle region (1-f < v1 < 1/(1-f))
        E_PV2_mu = integrate.dblquad(integrand_mu, 1-f, 1/(1-f), 
                                   lambda v1: 1/(v1*(1-f)), lambda v1: np.inf)[0]
        E_PV2_mm = integrate.dblquad(integrand_mm, 1-f, 1/(1-f),
                                   lambda v1: (1-f)/v1, lambda v1: 1/(v1*(1-f)))[0]
        E_PV2_md = integrate.dblquad(integrand_md, 1-f, 1/(1-f),
                                   lambda v1: 1e-4, lambda v1: (1-f)/v1)[0]
        
        # Lower region (v1 < 1-f)
        E_PV2_du = integrate.dblquad(integrand_du, 1e-4, 1-f,
                                   lambda v1: 1/(1-f)**2, lambda v1: np.inf)[0]
        E_PV2_dm = integrate.dblquad(integrand_dm, 1e-4, 1-f,
                                   lambda v1: 1, lambda v1: 1/(1-f)**2)[0]
        E_PV2_dd = integrate.dblquad(integrand_dd, 1e-4, 1-f,
                                   lambda v1: 1e-4, lambda v1: 1)[0]
        
        E_PV2 = (E_PV2_uu + E_PV2_um + E_PV2_ud + 
                 E_PV2_mu + E_PV2_mm + E_PV2_md +
                 E_PV2_du + E_PV2_dm + E_PV2_dd)
        
        return E_PV1, E_PV2 