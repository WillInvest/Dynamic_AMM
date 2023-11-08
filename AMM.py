import numpy as np
from solver import find_root_bisection
import math
from fee import NoFee, BaseFee
from utility_func import BaseUtility, ConstantProduct
from utils import add_dict, FeeDict, distribute_fees

from typing import Tuple, Dict, Callable, Literal


# Define class AMM
class AMM:

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
                 solver: Literal['bisec'] = 'bisec') -> None:
        
        if utility_func == "constant_product":
            self.utility_func = ConstantProduct(token_symbol='L')
        
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

        # This loops through each asset in the portfolio
        # Sets L = to the Nth root of the product
        # p = 1.
        # for key in self.portfolio:
        #     if key != 'L':
        #         p *= self.portfolio[key]
        # self.portfolio['L'] = p ** (1 / self.num_assets)
        
        num_l = self.utility_func.cal_liquid_token_amount(self.portfolio)
        self.portfolio['L'] = num_l
        
        self.lp_tokens = {'initial': num_l}
        
    def register_lp(self, user: str)-> None:
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
        return ret

    # @staticmethod
    # # Checks that each value in the portfolio aside from L is not 0
    # # Finds value based on log: Can Zhiyuan explain this?
    # def _utility_log(portfolio):
    #     p = 0.
    #     for key in portfolio:
    #         if key != 'L':
    #             assert portfolio[key] > 0, f"invalid value: {key}: {portfolio[key]}"
    #             p += np.log(portfolio[key])
    #     return p

    # @staticmethod
    # def _v_log(l, n):
    #     # Check both l and n are greater than 0
    #     assert l > 0 and n > 0, f'invalid value: l = {l}, n = {n}'
    #     # Return for what??
    #     return n * np.log(l)

    # @staticmethod
    # def _utility(portfolio: dict):
    #     # Return the value of all assets multiplied together
    #     # Not including L
    #     p = 1.
    #     for key in portfolio:
    #         if key != 'L':
    #             assert portfolio[key] > 0, f"invalid value: {key} - {portfolio[key]}"
    #             p *= portfolio[key]
    #     return p

    # @staticmethod
    # def _v(l, n):
    #     # Check both l and n are greater than 0
    #     assert l > 0 and n > 0, f'invalid value: l = {l}, n = {n}'
    #     return l ** n

    # def utility(self) -> float:
    #     # Calculate portfolio using log for easier calc
    #     # Convert to normal for readability
    #     return np.exp(self._utility_log(self.portfolio))

    def track_asset_ratio(self, asset_to_track, reference_asset):
        self.AfB.append(
            self.portfolio[asset_to_track] / self.portfolio[reference_asset])
        self.BfA.append(
            self.portfolio[reference_asset] / self.portfolio[asset_to_track])

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
        target = self.utility_func.target_function(full_portfolio=tmp_portfolio)
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

    def update_portfolio(self, *, delta_assets: dict={}, check: bool = True) -> Tuple[bool, dict]:
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
                if temp_result <= 0.: return False, {"error_info": f"Value: resulting non-positive amount of {k}."}
                self.portfolio[k] = temp_result

        except AssertionError:
            return False, {"error_info": f"AssertionError: {k} not in {delta_assets}."}
        return True, {}
    
    def update_fee(self, fees: dict) -> Tuple[bool, dict]:
        try:
            for keys in fees:
                assert keys in self.fees, f"Fee symbol {keys} is not legit."
                self.fees[keys] += fees[keys] # update fee portfolio
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
    
    def quote(self, s1: str, s2: str, s2_in: float) -> Tuple[float, Dict]:

        function_to_solve = self.helper_gen(s1, s2, s2_in)

        s1_in, _ = self.solver(
            function_to_solve, left_bound=-self.portfolio[s1] + 1)
        
        fee_dict = self.fee_structure.calculate_fee({s1: s1_in, s2: s2_in}, s2)
        info = {'asset_delta': {s1: s1_in, s2: s2_in}, 'fee': fee_dict}
        return s1_in, info

    

        
    # def detect_arbitrage(self, market_price):
    # # Assuming amm has asset pairs and tracks asset ratios
    # # Calculate implied prices for different asset pairs
    #     for i in range(len(self.AfB)):
    #         # Calculate implied price for A in terms of B
    #         implied_price_A_in_B = self.AfB[i] * market_price
    #         # Calculate implied price for B in terms of A
    #         implied_price_B_in_A = self.BfA[i] * (1 / market_price)

    #         # Check if there's an arbitrage opportunity
    #         if implied_price_A_in_B > round(1 / implied_price_B_in_A,2):
    #             print(f"Arbitrage Opportunity Detected!")
    #             print(f"Implied Price of A in terms of B: {implied_price_A_in_B}")
    #             print(f"Implied Price of B in terms of A: {implied_price_B_in_A}")
    #             print(f"Execute trades to take advantage of this opportunity")
    #             # You can place the necessary trade logic here
    #         else:
    #             print("No Arbitrage Opportunity Detected.")


    ## Function to set ratios to market value
    def set_market_trade(self, MP, inv1, inv2):
        inventory_1 = self.portfolio[inv1]
        
        inventory_2 = self.portfolio[inv2]
        
        ratio = inventory_1 / inventory_2

        if ratio > MP:
            y = math.sqrt(inventory_1 * inventory_2/MP) - inventory_2
            self.trade_swap(inv1,inv2,y)
            #print(f"This is your trade to execute: {inv2} {inv1} {y}")
            
        elif ratio < MP:
            x = math.sqrt(MP * inventory_1 *inventory_2) - inventory_1
            
            self.trade_swap(inv2,inv1,x)
            #print(f"This is your trade to execute: {inv1} {inv2} {x}")

    

class SimpleFeeAMM(AMM):
    def __init__(self, *, 
                 utility_func: Literal['constant_product'] = "constant_product", 
                 initial_portfolio: Dict[str, float] = None, 
                 initial_fee_portfolio: Dict[str, float] = None, 
                 ratio_denomination: str = "None", 
                 fee_structure: BaseFee = None, 
                 solver: Literal['bisec'] = 'bisec') -> None:
        super().__init__(utility_func=utility_func, 
                         initial_portfolio=initial_portfolio, 
                         initial_fee_portfolio=initial_fee_portfolio, 
                         ratio_denomination=ratio_denomination, 
                         fee_structure=fee_structure, 
                         solver=solver)
        
        
    def register_lp(self, user: str) -> None:
        assert sum(self.fees.values()) == 0, f"Must claim fees before registering a new liquidity provider."
        return super().register_lp(user)
    
    def _trade(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
        s1_in, info = self.quote(s1, s2, s2_in)
        fees = info['fee']
        updates = {s1: s1_in, s2: s2_in}
        success1, update_info1 = self.update_portfolio(delta_assets=updates, check=True)

        info['update_info_before_fee'] = update_info1
        if not success1: return False, info
        
        success2, update_info2 = self.update_fee(fees)
        info["update_info_fee"] = update_info2
        
        return success2, info
        
    def trade_swap(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
        if 'L' in (s1, s2):
            return False, {'error_info': f"Cannot update liqudity tokens using 'trade_swap'."}
        
        return self._trade(s1, s2, s2_in)
        
    def trade_liquidity(self, s1: str, s2: str, s2_in: float, lp_user: str) -> Tuple[bool, Dict]:
        if 'L' not in (s1, s2):
            return False, {'error_info': f"Must trade liquidity tokens using 'trade_liquidity'."}
        if not self.fees.is_empty():
            return False, {'error_info': f"Must claim fees before liquidity events."}
        if lp_user not in self.lp_tokens: 
            return False, {'error_info': f"Non-registered LP user: {lp_user}."}
        return self._trade(s1, s2, s2_in)
    
    def claim_fee(self):
        # ret = self.fees.copy()
        ret= distribute_fees(self.lp_tokens, self.fees)
        self.fees.reset()
        return ret
        
# class CompoundFeeAMM(AMM):
#     def __init__(self, *, 
#                  utility_func: Literal['constant_product'] = "constant_product", 
#                  initial_portfolio: Dict[str, float] = None, 
#                  initial_fee_portfolio: Dict[str, float] = None, 
#                  ratio_denomination: str = "None", 
#                  fee_structure: BaseFee = None, 
#                  solver: Literal['bisec'] = 'bisec') -> None:
#         super().__init__(utility_func=utility_func, 
#                          initial_portfolio=initial_portfolio, 
#                          initial_fee_portfolio=initial_fee_portfolio, 
#                          ratio_denomination=ratio_denomination, 
#                          fee_structure=fee_structure, 
#                          solver=solver)
        
#     def trade(self, s1: str, s2: str, s2_in: float) -> Tuple[bool, Dict]:
#         s1_in, info = self.quote(s1, s2, s2_in)
#         fees = info['fee']
#         updates = {s1: s1_in, s2: s2_in}
#         success1, update_info1 = self.update_portfolio(delta_assets=updates, check=True)
#         # success2, update_info2 = self.update_portfolio(delta_assets=fees, check=False)
        
#         info['update_info_before_fee'] = update_info1
#         # info['update_info_after_fee'] = update_info2
        
#         for keys in fees:
#             self.fees[keys] += fees[keys] # update fee portfolio
        
#         # return success1, info
    
#         if not (success1 or success2):
#             print(f"update success before fee: {success1}, after fee: {success2}")
#             print(info)
#             raise ValueError("illegal trade")
        
#         return success1 and success2, info


