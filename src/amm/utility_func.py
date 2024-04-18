# importing src directory
import sys
sys.path.append('..')
# library imports
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
# project imports
from amm.solver import find_root_bisection



CAL_ERROR_THRESHOLD = 1e-10

class BaseUtility(ABC):
    '''
    Base class of utility functions for Generalized AMM
    e.g., ConstantProduct = U(A, B)/V(L) = A*B/L^2
    '''
    def __init__(self, token_symbol: str) -> None:
        super().__init__()
        self.token_symbol = token_symbol
        
        
    def U(self, portfolio: dict, **kwargs) -> float:
        '''
        The utility value for given assets.
        '''
        return self._U(portfolio, **kwargs)
    
    def V(self, num_liqud_token: float, **kwargs) -> float:
        '''
        The V value for liquidity tokens
        '''
        return self._V(num_liqud_token, **kwargs)
    
    def U_log(self, portfolio: dict, **kwargs) -> float:
        '''
        The utility value for given assets, log of the U value
        '''
        return self._U_log(portfolio, **kwargs)
    
    def V_log(self, num_liqud_token: float, **kwargs) -> float:
        '''
        The V value for liquidity tokens,  log of the V value
        '''
        return self._V_log(num_liqud_token, **kwargs)
    
    def is_balance(self, full_portfolio: dict) -> bool:
        '''
        Check if the liquidity tokens are balanced with all other assets.
        Need the full inventory/portfolio as input. 
        '''
        return (self.target_function(full_portfolio) - 1) < CAL_ERROR_THRESHOLD
    
    def target_function(self, full_portfolio: dict) -> float:
        '''
        Calculate the target value U(.)/V(.) - 1
        should equal to 0.
        '''
        assert self.token_symbol in full_portfolio, f"Incomplete portfolio: missing liquidity token {self.token_symbol}"
        num_assets = len(full_portfolio) - 1
        target = self._U_log(
            full_portfolio) - self._V_log(full_portfolio[self.token_symbol], num_assets)
        return np.exp(target) - 1.
    
    def cal_liquid_token_amount(self, full_portfolio: dict) -> float:
        '''
        Calculate the needed amount of liquidity tokens to make the 
        portfolio balanced. Will ignore liquidity tokens in inputs. 
        '''
        copy_portfolio = full_portfolio.copy()
        def wrap_target(x):
            copy_portfolio[self.token_symbol] = x
            return self.target_function(copy_portfolio)
        result, _ = find_root_bisection(wrap_target, tolerance=CAL_ERROR_THRESHOLD, left_bound=1)
        copy_portfolio[self.token_symbol] = result
        
        assert self.is_balance(copy_portfolio), "Failed to calculate the amount of liquidity tokens."
        return result
        
        
    @abstractmethod
    def _U(self, portfolio: dict, **kwargs) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _V(self, num_liqud_token: float, **kwargs) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _U_log(self, portfolio: dict, **kwargs) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def _V_log(self, num_liqud_token: float, **kwargs) -> float:
        raise NotImplementedError
    
class ConstantProduct(BaseUtility):
    def __init__(self, token_symbol: str = 'L') -> None:
        super().__init__(token_symbol=token_symbol)
        
    def _U(self, portfolio: dict, **kwargs) -> float:
        # Return the value of all assets multiplied together
        # Not including L
        # p = 1.
        # for key in portfolio:
        #     if key != self.token_symbol:
        #         assert portfolio[key] > 0, f"invalid value: {key} - {portfolio[key]}"
        #         p *= portfolio[key]
        # return p
        return np.exp(self._U_log(portfolio))
    
    def _V(self, num_liqud_token: float, num_asset: int,  **kwargs) -> float:
        # l, n = num_liqud_token, num_asset
        # # Check both l and n are greater than 0
        # assert l > 0 and n > 0, f'invalid value: l = {l}, n = {n}'
        # return l ** n
        return np.exp(self._V_log(num_liqud_token, num_asset))
    
    def _U_log(self, portfolio: dict, **kwargs) -> float:
        p = 0.
        for key in portfolio:
            if key != self.token_symbol:
                assert portfolio[key] > 0, f"invalid value: {key}: {portfolio[key]}"
                p += np.log(portfolio[key])
        return p
    
    def _V_log(self, num_liqud_token: float, num_asset: int ,**kwargs) -> float:
        l, n = num_liqud_token, num_asset
        # Check both l and n are greater than 0
        assert l > 0 and n > 0, \
            f'invalid value: num_liqud_token = {l}, num_asset = {n}'
        # Return for log(V(L)) = log(L^n)
        return n * np.log(l)
    
