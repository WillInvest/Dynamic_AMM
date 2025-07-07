"""Calculation formulas for two-step AMM analysis"""
import numpy as np
from typing import Tuple, Optional
from scipy import integrate

class SwapReserves:
    """Class for lazy calculation of AMM values"""
    
    def __init__(self, ell_s: float, ell_r: float):
        self.ell_s = ell_s
        self.ell_r = ell_r
        
    def get_ls(self, f: float, v1_step: Tuple[float, str], v2_step: Optional[Tuple[float, str]] = None) -> Tuple[float, Optional[float]]:
        """
        Get stable token liquidity for one or two steps
        
        Args:
            f: Fee rate
            v1_step: Tuple of (v1, dir1) where v1 is price and dir1 is 'u', 'm', or 'd'
            v2_step: Optional tuple of (v2, dir2) for second step
            
        Returns:
            Tuple of (ls1, ls2) where ls2 is None if v2_step is None
        """
        v1, dir1 = v1_step
        
        # Calculate first step liquidity
        if dir1 == 'u':
            ls1 = self.ell_s * np.sqrt(v1*(1-f))
        elif dir1 == 'd':
            ls1 = self.ell_s * np.sqrt(v1/(1-f))
        else:  # dir1 == 'm'
            ls1 = self.ell_s
            
        # If no second step, return first step result only
        if v2_step is None:
            return ls1, None
            
        # Calculate second step liquidity
        v2, dir2 = v2_step
        if dir2 == 'u':
            ls2 = self.ell_s * np.sqrt(v1*v2*(1-f))
        elif dir2 == 'd':
            ls2 = self.ell_s * np.sqrt(v1*v2/(1-f))
        else:  # dir2 == 'm'
            ls2 = ls1
            
        return ls1, ls2

    def get_lr(self, f: float, v1_step: Tuple[float, str], v2_step: Optional[Tuple[float, str]] = None) -> Tuple[float, Optional[float]]:
        """
        Get risk token liquidity for one or two steps
        
        Args:
            f: Fee rate
            v1_step: Tuple of (v1, dir1) where v1 is price and dir1 is 'u', 'm', or 'd'
            v2_step: Optional tuple of (v2, dir2) for second step
            
        Returns:
            Tuple of (lr1, lr2) where lr2 is None if v2_step is None
        """
        v1, dir1 = v1_step
        
        # Calculate first step liquidity
        if dir1 == 'u':
            lr1 = self.ell_r * np.sqrt(1/((1-f)*v1))
        elif dir1 == 'd':
            lr1 = self.ell_r * np.sqrt((1-f)/v1)
        else:  # dir1 == 'm'
            lr1 = self.ell_r
            
        # If no second step, return first step result only
        if v2_step is None:
            return lr1, None
            
        # Calculate second step liquidity
        v2, dir2 = v2_step
        if dir2 == 'u':
            lr2 = self.ell_r * np.sqrt(1/((1-f)*v1*v2))
        elif dir2 == 'd':
            lr2 = self.ell_r * np.sqrt((1-f)/(v1*v2))
        else:  # dir2 == 'm'
            lr2 = lr1
            
        return lr1, lr2
    
    def get_delta_s(self, f: float, fee_source: str,
                    v1_step: Tuple[float, str],
                    v2_step: Optional[Tuple[float, str]] = None) -> Tuple[float, Optional[float]]:
        """
        Get stable token position change for one or two steps
        
        Args:
            f: Fee rate
            fee_source: 'in' for incoming fees, 'out' for outgoing fees
            v1_step: Tuple of (v1, dir1) where v1 is price and dir1 is 'u', 'm', or 'd'
            v2_step: Optional tuple of (v2, dir2) for second step
            
        Returns:
            Tuple of (delta_s1, delta_s2) where delta_s2 is None if v2_step is None
        """
        v1, dir1 = v1_step
        
        # Calculate first step delta_s
        if fee_source == 'in':
            if dir1 == 'u':
                delta_s1 = self.ell_s * (np.sqrt(v1/(1-f)) - 1/(1-f))
            elif dir1 == 'd':
                delta_s1 = self.ell_s * (1 - np.sqrt(v1/(1-f)))
            else:  # dir1 == 'm'
                delta_s1 = 0.0
        else:  # fee_source == 'out'
            if dir1 == 'u':
                delta_s1 = self.ell_s * (np.sqrt(v1*(1-f)) - 1)
            elif dir1 == 'd':
                delta_s1 = self.ell_s * (1 - np.sqrt(v1/(1-f)))
            else:  # dir1 == 'm'
                delta_s1 = 0.0
            
        # If no second step, return first step result only
        if v2_step is None:
            return delta_s1, None
        
        # Calculate second step delta_s
        v2, dir2 = v2_step
        
        if fee_source == 'in':
            if dir1 == 'u':
                if dir2 == 'u':
                    delta_s2 = self.ell_s * (np.sqrt(v1*v2/(1-f)) - np.sqrt(v1/(1-f)))
                elif dir2 == 'd':
                    delta_s2 = self.ell_s * (np.sqrt(v1*(1-f)) - np.sqrt(v1*v2/(1-f)))
                else:  # dir2 == 'm'
                    delta_s2 = 0.0
            elif dir1 == 'm':
                if dir2 == 'u':
                    delta_s2 = self.ell_s * (np.sqrt(v1*v2/(1-f)) - 1/(1-f))
                elif dir2 == 'd':
                    delta_s2 = self.ell_s * (1 - np.sqrt(v1*v2/(1-f)))
                else:  # dir2 == 'm'
                    delta_s2 = 0.0
            else:  # dir1 == 'd'
                if dir2 == 'u':
                    delta_s2 = self.ell_s * (np.sqrt(v1*v2/(1-f)) - np.sqrt(v1/(1-f)**3))
                elif dir2 == 'd':
                    delta_s2 = self.ell_s * (np.sqrt(v1/(1-f)) - np.sqrt(v1*v2/(1-f)))
                else:  # dir2 == 'm'
                    delta_s2 = 0.0
        else:  # fee_source == 'out'
            if dir1 == 'u':
                if dir2 == 'u':
                    delta_s2 = self.ell_s * (np.sqrt(v1*v2*(1-f)) - np.sqrt(v1*(1-f)))
                elif dir2 == 'd':
                    delta_s2 = self.ell_s * (np.sqrt(v1*(1-f)) - np.sqrt(v1*v2/(1-f)))
                else:  # dir2 == 'm'
                    delta_s2 = 0.0
            elif dir1 == 'm':
                if dir2 == 'u':
                    delta_s2 = self.ell_s * (np.sqrt(v1*v2*(1-f)) - 1)
                elif dir2 == 'd':
                    delta_s2 = self.ell_s * (1 - np.sqrt(v1*v2/(1-f)))
                else:  # dir2 == 'm'
                    delta_s2 = 0.0
            else:  # dir1 == 'd'
                if dir2 == 'u':
                    delta_s2 = self.ell_s * (np.sqrt(v1*v2*(1-f)) - np.sqrt(v1/(1-f)))
                elif dir2 == 'd':
                    delta_s2 = self.ell_s * (np.sqrt(v1/(1-f)) - np.sqrt(v1*v2/(1-f)))
                else:  # dir2 == 'm'
                    delta_s2 = 0.0
                
        return delta_s1, delta_s2

    def get_delta_r(self, f: float, fee_source: str,
                    v1_step: Tuple[float, str],
                    v2_step: Optional[Tuple[float, str]] = None) -> Tuple[float, Optional[float]]:
        """
        Get risk token position change for one or two steps
        
        Args:
            f: Fee rate
            fee_source: 'in' for incoming fees, 'out' for outgoing fees
            v1_step: Tuple of (v1, dir1) where v1 is price and dir1 is 'u', 'm', or 'd'
            v2_step: Optional tuple of (v2, dir2) for second step
            
        Returns:
            Tuple of (delta_r1, delta_r2) where delta_r2 is None if v2_step is None
        """
        v1, dir1 = v1_step
        
        # Calculate first step delta_r
        if fee_source == 'in':
            if dir1 == 'u':
                delta_r1 = self.ell_r * (1 - np.sqrt(1/(v1*(1-f))))
            elif dir1 == 'd':
                delta_r1 = self.ell_r * (np.sqrt(1/(v1*(1-f))) - 1/(1-f))
            else:  # dir1 == 'm'
                delta_r1 = 0.0
        else:  # fee_source == 'out'
            if dir1 == 'u':
                delta_r1 = self.ell_r * (1 - np.sqrt(1/(v1*(1-f))))
            elif dir1 == 'd':
                delta_r1 = self.ell_r * (np.sqrt((1-f)/v1) - 1)
            else:  # dir1 == 'm'
                delta_r1 = 0.0
            
        # If no second step, return first step result only
        if v2_step is None:
            return delta_r1, None
        
        # Calculate second step delta_r
        v2, dir2 = v2_step
        
        if fee_source == 'in':
            if dir1 == 'u':
                if dir2 == 'u':
                    delta_r2 = self.ell_r * (np.sqrt(1/(v1*(1-f))) - np.sqrt(1/(v1*v2*(1-f))))
                elif dir2 == 'd':
                    delta_r2 = self.ell_r * (np.sqrt(1/(v1*v2*(1-f))) - np.sqrt(1/(v1*(1-f)**3)))
                else:  # dir2 == 'm'
                    delta_r2 = 0.0
            elif dir1 == 'm':
                if dir2 == 'u':
                    delta_r2 = self.ell_r * (1 - np.sqrt(1/(v1*v2*(1-f))))
                elif dir2 == 'd':
                    delta_r2 = self.ell_r * (np.sqrt(1/(v1*v2*(1-f))) - 1/(1-f))
                else:  # dir2 == 'm'
                    delta_r2 = 0.0
            else:  # dir1 == 'd'
                if dir2 == 'u':
                    delta_r2 = self.ell_r * (np.sqrt((1-f)/(v1)) - np.sqrt(1/(v1*v2*(1-f))))
                elif dir2 == 'd':
                    delta_r2 = self.ell_r * (np.sqrt(1/(v1*v2*(1-f))) - np.sqrt(1/(v1*(1-f))))
                else:  # dir2 == 'm'
                    delta_r2 = 0.0
        else:  # fee_source == 'out'
            if dir1 == 'u':
                if dir2 == 'u':
                    delta_r2 = self.ell_r * (np.sqrt(1/(v1*(1-f))) - np.sqrt(1/(v1*v2*(1-f))))
                elif dir2 == 'd':
                    delta_r2 = self.ell_r * (np.sqrt((1-f)/(v1*v2)) - np.sqrt(1/(v1*(1-f))))
                else:  # dir2 == 'm'
                    delta_r2 = 0.0
            elif dir1 == 'm':
                if dir2 == 'u':
                    delta_r2 = self.ell_r * (1 - np.sqrt(1/(v1*v2*(1-f))))
                elif dir2 == 'd':
                    delta_r2 = self.ell_r * (np.sqrt((1-f)/(v1*v2)) - 1)
                else:  # dir2 == 'm'
                    delta_r2 = 0.0
            else:  # dir1 == 'd'
                if dir2 == 'u':
                    delta_r2 = self.ell_r * (np.sqrt((1-f)/v1) - np.sqrt(1/(v1*v2*(1-f))))
                elif dir2 == 'd':
                    delta_r2 = self.ell_r * (np.sqrt((1-f)/(v1*v2)) - np.sqrt((1-f)/v1))
                else:  # dir2 == 'm'
                    delta_r2 = 0.0
                
        return delta_r1, delta_r2