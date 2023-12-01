import numpy as np
from solver import find_root_bisection
import math
from fee import NoFee, BaseFee
from utility_func import BaseUtility, ConstantProduct
from utils import add_dict, FeeDict, distribute_fees, add_lp_tokens

from typing import Tuple, Dict, Callable, Literal
from abc import ABC, abstractmethod
from threading import Lock


# Define class AMM
class AMM(ABC):

    # Initalize default values of A and B in portfolio
    default_init_portfolio = {'A': 1000.0, 'B': 10000.0, "L": None}
    fee_init_portfolio = {'A': 0.0, 'B': 0.0, "L": 0.0}

    # Create a new AMM object
    # set the initial portfolio when creating the object
    def __init__(self, *,
                 utility_func: Literal["constant_product"] = "constant_product",
                 initial_portfolio: Dict[str, float] = None,
                 initial_fee_portfolio: Dict[str, float] = None,
                 ratio_denomination: str = "None",
                 fee_structure: BaseFee = None,
                 fee_precharge: bool = True,
                 solver: Literal['bisec'] = 'bisec') -> None:

        if utility_func == "constant_product":
            self.utility_func = ConstantProduct(token_symbol='L')
<<<<<<< HEAD
            
        self.fee_precharge = fee_precharge
        
=======

>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
        # Set solver
        if solver == 'bisec':
            self.solver = find_root_bisection
        else:
            raise RuntimeError(f"No solver found for {solver}.")

        # Set fee structure
        if fee_structure is not None:
            self.fee_structure = fee_structure
        else:
            self.fee_structure = NoFee()

        # Set the portfolio if initial portfolio is given
        if initial_portfolio:
            self.portfolio = initial_portfolio
        # Set the portfolio to default portfolio otherwise
        else:
            self.portfolio = self.default_init_portfolio.copy()

        # Set the fee portfolio if fee portfolio is given
        if initial_fee_portfolio:
            self.fees = initial_fee_portfolio
        # Set the fee portfolio to default fee portfolio otherwise
        else:
            self.fees = self.fee_init_portfolio.copy()
        self.fees = FeeDict(self.fees)

        # Create a list to store asset ratios
        self.AfB = []
        self.BfA = []

        # Set the ratio denomination if given
        if ratio_denomination:
            self.denomination = ratio_denomination
        else:
            # set to first value in portfolio
            self.denomination = [self.portfolio.keys()][0]

        # Number of assets excluding L, ie A, B -> 2
        self.num_assets = len(self.portfolio) - 1

        num_l = self.utility_func.cal_liquid_token_amount(self.portfolio)
        self.portfolio['L'] = num_l

        self.lp_tokens = {'initial': num_l}
<<<<<<< HEAD
        
        self.lock = Lock()
        
=======

>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
    def curr_utility(self) -> float:
        return self.utility_func.U(self.portfolio)

    def register_lp(self, user: str) -> None:
        '''
        Register a liquidity provider entry. 
        '''

        if user in self.lp_tokens:
            print(f"User {user} is already in LP list.")
        else:
            self.lp_tokens[user] = 0.

    # Create list to print AMM assets
    def __repr__(self) -> str:
        ret = '-' * 20 + '\n'
        # add each asset to the list
        for key in self.portfolio:
            ret += f"{key}: {self.portfolio[key]}\n"
        ret += '-' * 20 + '\n'
        # add each fee to the list
        for fee in self.fees:
            ret += f"F{fee}: {self.fees[fee]}\n"
        ret += '-' * 20 + '\n'
        ret += f"Utility: {self.curr_utility()}\n"
        ret += '-' * 20 + '\n'
        return ret

    def track_asset_ratio(self, asset_to_track, reference_asset):
        self.AfB.append(
            self.portfolio[asset_to_track] / self.portfolio[reference_asset])
        self.BfA.append(
            self.portfolio[reference_asset] / self.portfolio[asset_to_track])
    
    def asset_ratio(self, asset_to_track, reference_asset):
        return self.portfolio[asset_to_track] / self.portfolio[reference_asset]


    def target_function(self, *, delta_assets: dict = {}) -> float:
        '''
        Calculate the target value with a change of inventories
        '''
        tmp_portfolio = self.portfolio.copy()
        # Check for change in asset
        for asset in tmp_portfolio:
            tmp_portfolio[asset] += delta_assets.get(asset, 0.)

        # --------- regular calculation -----------
        # target = self._utility(tmp_portfolio)/self._v(tmp_portfolio['L'], self.num_assets);
        # print(target, self._utility(tmp_portfolio), self._v(tmp_portfolio['L'], self.num_assets))
        # return target-1.
        # ----------------- end -------------------

        # --------- log calculation -----------
        target = self.utility_func.target_function(
            full_portfolio=tmp_portfolio)
        # ----------------- end -------------------
        return np.exp(target) - 1.

    def get_cummulative_fees(self) -> float:
        total_fees = 0
        for key, value in self.fees.items():
            if key == 'L':
                pass
            elif key == self.denomination:
                total_fees += value
            else:
                total_fees += (self.fees[key] /
                               self.fees[self.denomination])
        return total_fees

    def update_portfolio(self, *, delta_assets: dict = {}, check: bool = True) -> Tuple[bool, dict]:
        '''
        Manually update portfolio, this may lead to a unbalanced portfolio. 
        '''
        if check:
            target = self.target_function(delta_assets=delta_assets)
            assert abs(target) < 1e-8, f"Target: {target}"
        try:
            for k in delta_assets:
                assert k in self.portfolio, k
                temp_result = self.portfolio[k] + delta_assets[k]
                if temp_result <= 0.:
                    return False, {"error_info": f"Value: resulting non-positive amount of {k}."}
                self.portfolio[k] = temp_result

        except AssertionError:
            return False, {"error_info": f"AssertionError: {k} not in {delta_assets}."}
        return True, {}

    def update_fee(self, fees: dict) -> Tuple[bool, dict]:
        try:
            for keys in fees:
                assert keys in self.fees, f"Fee symbol {keys} is not legit."
<<<<<<< HEAD
                self.fees[keys] += fees[keys]# update fee portfolio
=======
                self.fees[keys] += fees[keys]  # update fee portfolio
>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
        except AssertionError as info:
            return False, {'error_info': info}
        return True, {}

    def helper_gen(self, s1: str, s2: str, s2_in: float) -> Callable[[float], float]:
        # Calculates target value of changed value
        assert s1 in self.portfolio and s2 in self.portfolio, f"{s1} or {s2} not in portfolio"
        assert self.portfolio[s2] + s2_in > 0., f"No enough assets/tokens"
        assert s1 != s2, f"same s1 and s2 input"

        def func(x: float) -> float:
            delta_assets = {s1: x, s2: s2_in}
            return self.target_function(delta_assets=delta_assets)

        return func

    def _quote_pre_fee(self, s1: str, s2: str, s2_in: float) -> Tuple[float, Dict]:
        # assert fee_asset in (s1, s2), f"Illegal fee asset: {fee_asset} for transaction between {s1} and {s2}."
<<<<<<< HEAD
        '''
        Calculate the fee in unit of s2
        '''
        fee_dict = self.fee_structure.calculate_fee({s1: None, s2: s2_in}, s2, amm = self)
=======

        fee_dict = self.fee_structure.calculate_fee(
            {s1: None, s2: s2_in}, s2, portfolio=self.portfolio, amm=self)
>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
        actual_s2_in = s2_in - fee_dict[s2]

        s1_in, info = self._quote_no_fee(s1, s2, actual_s2_in)

<<<<<<< HEAD
        info.update({'asset_delta': {s1: s1_in, s2: actual_s2_in}, 'fee': fee_dict})
        return s1_in, info 
    
    def _quote_post_fee(self, s1: str, s2: str, s2_in: float) -> Tuple[float, Dict]:  
        '''
        Calculate the fee in unit of s1
        
        actual_s1_in=(s1_in+s1_fee)->amm->s2_in
        '''

        s1_in, info = self._quote_no_fee(s1, s2, s2_in)
        
        fee_dict = self.fee_structure.calculate_fee({s1: s1_in, s2: s2_in}, s1, amm = self)
=======
        info.update(
            {'asset_delta': {s1: s1_in, s2: actual_s2_in}, 'fee': fee_dict})
        return s1_in, info

    def _quote_post_fee(self, s1: str, s2: str, s2_in: float) -> Tuple[float, Dict]:

        s1_in, info = self._quote_no_fee(s1, s2, s2_in)

        fee_dict = self.fee_structure.calculate_fee(
            {s1: s1_in, s2: s2_in}, s1, portfolio=self.portfolio, amm=self)

>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
        actual_s1_in = s1_in + fee_dict[s1]

        info.update({'asset_delta': {s1: s1_in, s2: s2_in}, 'fee': fee_dict})
        return actual_s1_in, info

# asset_out, _ = amm._quote_no_fee(
#                 receive_asset, fee_asset, transaction_dict[fee_asset])

    def _quote_no_fee(self, s1: str, s2: str, s2_in: float) -> Tuple[float, Dict]:
        # assert fee_asset in (s1, s2), f"Illegal fee asset: {fee_asset} for transaction between {s1} and {s2}."

        function_to_solve = self.helper_gen(s1, s2, s2_in)

        s1_in, _ = self.solver(
            function_to_solve, left_bound=-self.portfolio[s1] + 1)

        info = {'asset_delta': {s1: s1_in, s2: s2_in}, 'fee': {}}
        return s1_in, info

    def quote(self, s1: str, s2: str, s2_in: float) -> Tuple[float, Dict]:
        is_liquidity_event = ('L' in (s1, s2))
<<<<<<< HEAD
        if not is_liquidity_event: # swap
            # if s2_in >= 0 and False: 
            #     return self._quote_pre_fee(s1, s2, s2_in)
            # else:
            if self.fee_precharge:
=======
        if not is_liquidity_event:  # swap
            if s2_in >= 0:
                return self._quote_pre_fee(s1, s2, s2_in)
            else:
>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
                return self._quote_post_fee(s1, s2, s2_in)
            else:
                return self._quote_pre_fee(s1, s2, s2_in)
        else:
            return self._quote_no_fee(s1, s2, s2_in)

    def trade_swap(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
        '''
        The function should only do swaps.
        '''
<<<<<<< HEAD
        if 'L' in (s1, s2):
            return False, {'error_info': f"Cannot update liqudity tokens using 'trade_swap'."}
        self.lock.acquire()
        ret = self._trade(s1, s2, s2_in)
        self.lock.release()
        return ret
        
=======

        if 'L' in (s1, s2):
            return False, {'error_info': f"Cannot update liqudity tokens using 'trade_swap'."}

        return self._trade(s1, s2, s2_in)

>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
    def trade_liquidity(self, s1: str, s2: str, s2_in: float, lp_user: str) -> Tuple[bool, Dict]:
        '''
        The function allows the registered LP users
        to trade liquidity tokens. 
        '''
        
        if 'L' not in (s1, s2):
            return False, {'error_info': f"Must trade liquidity tokens using 'trade_liquidity'."}
        if lp_user not in self.lp_tokens:
            return False, {'error_info': f"Non-registered LP user: {lp_user}."}
<<<<<<< HEAD
        self.lock.acquire()
        ret = self._trade(s1, s2, s2_in)
        self.lock.release()
        return ret
    
=======
        return self._trade(s1, s2, s2_in)

>>>>>>> 0222cca (updated triangle fee and passed amm through fee calls so can solve within triangle -- its currently producing negative fees, so i am in the process of figuring out where in the integral calculation this is arising or if its as simple as abs() the output)
    @abstractmethod
    def _trade(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
        raise NotImplementedError


class SimpleFeeAMM(AMM):

    def register_lp(self, user: str) -> None:
        assert sum(self.fees.values(
        )) == 0, f"Must claim fees before registering a new liquidity provider."
        return super().register_lp(user)

        #     succ, info = amm.trade_swap(s1, s2, s2_in)
        # if succ:
        #     print(f"User pay {s1}: {info['pay_s1']}")

    def _trade(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
        s1_in, info = self.quote(s1, s2, s2_in)
        info['pay_s1'] = s1_in
        fees = info['fee']
        updates = info['asset_delta']

        print("_trade", s1_in, info)

        success1, update_info1 = self.update_portfolio(
            delta_assets=updates, check=True)

        info['update_info_before_fee'] = update_info1
        if not success1:
            return False, info

        success2, update_info2 = self.update_fee(fees)
        info["update_info_fee"] = update_info2

        print("_trade", success1, success2, info)
        return success2, info

    def trade_liquidity(self, s1: str, s2: str, s2_in: float, lp_user: str) -> Tuple[bool, Dict]:
        if not self.fees.is_empty():
            return False, {'error_info': f"Must claim fees before liquidity events."}
        return super().trade_liquidity(s1, s2, s2_in, lp_user)

    def claim_fee(self):
        # ret = self.fees.copy()
        ret = distribute_fees(self.lp_tokens, self.fees)
        self.fees.reset()
        return ret

# class CompoundFeeAMM(AMM):

#     def _trade(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
#         s1_in, info = self.quote(s1, s2, s2_in)
#         fees = info['fee']
#         updates = {s1: s1_in, s2: s2_in}
#         success1, update_info1 = self.update_portfolio(delta_assets=updates, check=True)

#         assert len(fees) == 1, f"Fees in more than one asset in one transaction is not supported {fees}. "

#         fees.keys()
#         info['update_info_before_fee'] = update_info1

#         if not (success1 or success2):
#             print(f"update success before fee: {success1}, after fee: {success2}")
#             print(info)
#             raise ValueError("illegal trade")

#         return success1 and success2, info
