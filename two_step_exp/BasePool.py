"""Base class for AMM pool calculations with cached phi values"""
import numpy as np
from typing import Dict, Tuple
from functools import lru_cache
from SwapReserve import SwapReserves

class BasePool:
    """Base class for AMM pool calculations"""
    
    def __init__(self, ell_s: float, ell_r: float):
        """
        Initialize BasePool
        
        Args:
            ell_s: Stable token liquidity initial reserve
            ell_r: Risk token liquidity initial reserve
        """
        self.ell_s = ell_s
        self.ell_r = ell_r
        self.calc = SwapReserves(ell_s, ell_r)
        self._phi_cache: Dict[Tuple[float, float], float] = {}
        
    @lru_cache(maxsize=10000)
    def phi(self, v: float, sigma: float) -> float:
        """
        Cached PDF for geometric brownian motion with zero drift
        
        Args:
            v: Price ratio
            sigma: Volatility
            
        Returns:
            PDF value
        """
        return (1/(v * sigma * np.sqrt(2*np.pi)) * 
                np.exp(-(np.log(v) + sigma**2/2)**2/(2*sigma**2))) 