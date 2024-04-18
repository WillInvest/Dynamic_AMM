# importing src directory
import sys
sys.path.append('..')
# general libraries
import numpy as np
import pandas as pd
import math
from typing import Tuple, Dict, Callable, Literal
from abc import ABC, abstractmethod
# project libraries
from amm.solver import find_root_bisection
from amm.fee import NoFee, BaseFee
from amm.utility_func import BaseUtility, ConstantProduct
from amm.utils import add_dict, FeeDict, distribute_fees, add_lp_tokens

# ABSTRACT CLASS
class AMM(ABC):
    # DEFAULT INVENTORY
    default_init_portfolio = {'A': 10000.0, 'B': 10000.0, "L": None}
    fee_init_portfolio = {'A': 0.0, 'B': 0.0, "L": 0.0}
    # MARKET TRACKING
    market_init_portfolio = {'A': None, 'B': None, "L": None}

    # INITIALIZING AMM
    def __init__(self, *,
                 utility_func: Literal["constant_product"] = "constant_product",
                 initial_portfolio: Dict[str, float] = None,
                 initial_fee_portfolio: Dict[str, float] = None,
                 market_init_portfolio: Dict[str, float] = None,
                 market_data: pd.DataFrame = None,
                 fee_structure: BaseFee = None,
                 solver: Literal['bisec'] = 'bisec') -> None:
        
    # INITITALIZING AMM FUNCTIONALITY
        # # set utility function - constant product
        if utility_func == "constant_product":
            self.utility_func = ConstantProduct(token_symbol='L')
        # # set solver - bisection
        if solver == 'bisec':
            self.solver = find_root_bisection
        else:
            raise RuntimeError(f"No solver found for {solver}.")
        # # set fee structure - default to no fee
        if fee_structure is not None:
            self.fee_structure = fee_structure
        else:
            self.fee_structure = NoFee()
        
    # INITITALIZING AMM INVENTORY
        # # set initial portfolio if given
        if initial_portfolio:
            self.portfolio = initial_portfolio
        # # otherwise set to default
        else:
            self.portfolio = self.default_init_portfolio.copy()
        # # set initial fee portfolio if given
        
        if initial_fee_portfolio:
            self.fees = initial_fee_portfolio
        # # otherwise set to default
        else:
            self.fees = self.fee_init_portfolio.copy()
        # # create fee dictionary
        self.fees = FeeDict(self.fees)

        # # set initial market portfolio if given
        if market_init_portfolio:
            self.market_data = market_init_portfolio
        # # otherwise set to default
        else:
            self.market_data = self.market_init_portfolio.copy()

    # CREATE AMM DATA STORAGE
        # # define df to store AMM data
        if market_data:
            self.market_data = market_data
        # # if not given scheme, create new df w/ port & fee scheme
        else:
            columns = [key for key in self.portfolio] + [f"F_{key}" for key in self.fees]
            self.data = pd.DataFrame(columns = columns)

    # INITIALIZE AMM DATA & LIQUIDITY TOKENS
        # # save number of assets (excluding L)
        self.num_assets = len(self.portfolio) - 1
        # # use utility function to initialize liquidity tokens minted
        init_num_lp = self.utility_func.cal_liquid_token_amount(self.portfolio)
        self.portfolio['L'] = init_num_lp
        # # create liquidity token info
        self.lp_tokens = {'initial': init_num_lp}
        # # add LP users later if implementing (e.g. 'user_id': 1.0)

    # ADD LIQUIDITY PROVIDER TO AMM
    def register_lp(self, user: str) -> None:
        '''
        register a liquidity provider w/ empty balance
        user: str - user id
        '''
        # add user to LP list if not already in
        if user in self.lp_tokens: print(f"User {user} is already in LP list.")
        else: self.lp_tokens[user] = 0.

    # RETURN AMM STRING OUTPUT
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

    def target_function(self, *, delta_assets: dict = {}) -> float:
        '''
        calculate the target value with a change of inventories
        delta_assets: dict - change in assets (same format as portfolio)
        '''
        tmp_portfolio = self.portfolio.copy()
        # check for change in asset
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
        return self.fees

# UPDATE AMM INVENTORY
    def update_portfolio(self, *, delta_assets: dict = {}, check: bool = True) -> Tuple[bool, dict]:
        '''
        Manually update portfolio, this may lead to a unbalanced portfolio. Take completed trade dictionary, delta_assets, and update the amm portfolio accordingly
        delta_assets: dict - trade dictionary
        check: 
# TODO: figure out why it may be unbalanced and what check is
        '''
        if check:
            target = self.target_function(delta_assets=delta_assets) 
            assert abs(target) < 1e-8, f"Target: {target}"
        try:
            # for each asset in the swap
            for k in delta_assets:
                # check trade asset is in portfolio
                assert k in self.portfolio, k
                # temp store updated amm value & check
                temp_result = self.portfolio[k] + delta_assets[k]
                # check for negative amount
                if temp_result <= 0.: return False, {"error_info": f"Value: resulting non-positive amount of {k}."}
                # update portfolio value
                self.portfolio[k] = temp_result
        except AssertionError: return False, {"error_info": f"AssertionError: {k} not in {delta_assets}."}
        return True, {} # return success

# UPDATE FEE INVENTORY
    def update_fee(self, fees: dict) -> Tuple[bool, dict]:
        """
        take fee dictionary, fees,  and update the amm fee portfolio accordingly
        fees: dict - fee dictionary
        returns: bool - success or failure, dict - empty dict
        """
        try: # catch assertion error
            for keys in fees: # iterate through fee dictionary
                assert keys in self.fees, f"Fee symbol {keys} is not in the pool."
                self.fees[keys] += fees[keys]  # update fee portfolio for each in fee dictionary
        except AssertionError as info: return False, {'error_info': info} # catch assertion error
        return True, {} # return success

# TBD 
    def helper_gen(self, s1: str, s2: str, s2_in: float) -> Callable[[float], float]:
        # Calculates target value of changed value
        assert s1 in self.portfolio and s2 in self.portfolio, f"{s1} or {s2} not in portfolio"
        assert self.portfolio[s2] + s2_in > 0., f"No enough assets/tokens"
        assert s1 != s2, f"same s1 and s2 input"

        def func(x: float) -> float:
            delta_assets = {s1: x, s2: s2_in}
            return self.target_function(delta_assets=delta_assets)

        return func

# QUOTE W/ FEE ON ASSET IN
    def _quote_pre_fee(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[float, Dict]:
        fee_dict = self.fee_structure.calculate_fee({asset_out: None, asset_in: asset_in_n}, asset_in, portfolio=self.portfolio, amm=self) # calc fee 
        actual_in_n = asset_in_n - fee_dict[asset_in] # calc how much goes into public amm inventory pool, less fees
        asset_out_n, info = self._quote_no_fee(asset_out, asset_in, actual_in_n) # calc how much out w/ fee already deducted
        info.update({'asset_delta': {asset_out: asset_out_n, asset_in: actual_in_n}, 'fee': fee_dict}) # add adjusted inventory & fees
        return asset_out_n, info

# QUOTE W/ FEE ON ASSET OUT
    def _quote_post_fee(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[float, Dict]:
        asset_out_n, info = self._quote_no_fee(asset_out, asset_in, asset_in_n) # get asset out amount
        fee_dict = self.fee_structure.calculate_fee({asset_out: asset_out_n, asset_in: asset_in_n}, asset_out, portfolio=self.portfolio, amm=self) # calc fee
        actual_out_n = asset_out_n + fee_dict[asset_out] # add back fee that is seperated in swap
        info.update({'asset_delta': {asset_out: asset_out_n, asset_in: asset_in_n}, 'fee': fee_dict}) # updated inventory & save seperated fee 
        return actual_out_n, info

# QUOTE AMOUNT OUT W/O FEE ON EITHER ASSET
    def _quote_no_fee(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[float, Dict]:
        function_to_solve = self.helper_gen(asset_out, asset_in, asset_in_n) # get target function
        asset_out_n, _ = self.solver(function_to_solve, left_bound=-self.portfolio[asset_out] + 1) # solve for asset out amount
        info = {'asset_delta': {asset_out: asset_out_n, asset_in: asset_in_n}, 'fee': {}} # store trade info
        return asset_out_n, info # return asset out amount & trade info

# DIRECT QUOTE REQUESTS (LIQUIDITY, WHICH ASSET TO FEE)
    def quote(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[float, Dict]:
        """
        quote the amount of asset_out received for a given amount, asset_in_n, of asset_in
        asset_out: str - asset to receive
        asset_in: str - asset to pay
        asset_in_n: float - amount of asset_in to pay (positive sign for paying fee on input asset, negative sign for paying fee on output asset)
        """
        is_liquidity_event = ('L' in (asset_out, asset_in)) # check if liquidity event
        if not is_liquidity_event:  # if not liquidity event
            if asset_in_n >= 0: return self._quote_pre_fee(asset_out, asset_in, asset_in_n) # quote, fee according to sign:
            else: return self._quote_post_fee(asset_out, asset_in, asset_in_n) # use + / - symbol to determine pre or post
        else: return self._quote_no_fee(asset_out, asset_in, asset_in_n) # no fee if liquidity event

# CALL AMM'S TRADE FUNCTION
    def trade_swap(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[bool, Dict]:
        '''
        The function should only do swaps.
        asset_out: str - asset to receive
        asset_in: str - asset to pay
        asset_in_n: float - amount of asset_in to pay (positive sign for paying fee on input asset, negative sign for paying fee on output asset)
        '''
        if asset_in_n == 0.0:
#NOTE added fee dict to this bcs _trade usually does, but sean didnt have so shouldnt create bug but just temp note here
            return False, {'asset_delta': {"A": 0.0, "B": 0.0}, 'fee': {"FA": 0.0, "FB": 0.0}}
        if 'L' in (asset_out, asset_in): return False, {'error_info': f"Cannot update liqudity tokens using 'trade_swap'."}
        return self._trade(asset_out, asset_in, asset_in_n) # if not liquidity event, call trade function

# SWAP LP TOKENS
    def trade_liquidity(self, asset_out: str, asset_in: str, asset_in_n: float, lp_user: str) -> Tuple[bool, Dict]:
        '''
        The function allows the registered LP users
        to trade liquidity tokens. 
        asset_out: str - asset to receive
        asset_in: str - asset to pay
        asset_in_n: float - amount of asset_in to pay (positive sign for paying fee on input asset, negative sign for paying fee on output asset)
        lp_user: str - LP user id
        '''
        if 'L' not in (asset_out, asset_in):
            return False, {'error_info': f"Must trade liquidity tokens using 'trade_liquidity'."}
        if lp_user not in self.lp_tokens:
            return False, {'error_info': f"Non-registered LP user: {lp_user}."}
        return self._trade(asset_out, asset_in, asset_in_n)

# WRAPPER FOR AMM'S TRADE FUNCTION
    @abstractmethod
    def _trade(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[bool, Dict]:
        raise NotImplementedError

class SimpleFeeAMM(AMM):
# TBD????
    def register_lp(self, user: str) -> None:
        assert sum(self.fees.values(
        )) == 0, f"Must claim fees before registering a new liquidity provider."
        return super().register_lp(user)
        #     succ, info = amm.trade_swap(s1, s2, s2_in)
        # if succ:
        #     print(f"User pay {s1}: {info['pay_s1']}")

# SIMPLE CLASS TRADE FUNCTION
    def _trade(self, asset_out: str, asset_in: str, asset_in_n: float) -> Tuple[bool, Dict]:
        # find the amount of asset_out & collect the trade info
        asset_out_n, info = self.quote(asset_out, asset_in, asset_in_n)
        fees = info['fee'] # set fees to fee dictionary
        updates = info['asset_delta'] # set updates for call
        success1, update_info1 = self.update_portfolio(delta_assets=updates, check=True) # update amm portfolio according to trade
        if not success1: return False, info # check for success
        success2, update_info2 = self.update_fee(fees) # update amm fees according to trade
        return success2, info # return completion message & trade info

# TBD
    def trade_liquidity(self, asset_out: str, asset_in: str, asset_in_n: float, lp_user: str) -> Tuple[bool, Dict]:
        if not self.fees.is_empty(): return False, {'error_info': f"Must claim fees before liquidity events."}
        return super().trade_liquidity(asset_out, asset_in, asset_in_n, lp_user)

# TBD
    def claim_fee(self):
        # ret = self.fees.copy()
        ret = distribute_fees(self.lp_tokens, self.fees)
        self.fees.reset()
        return ret

