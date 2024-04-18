# importing src directory
import sys
sys.path.append('..')
# library imports
import os
from abc import ABC, abstractmethod
from typing import Literal, Dict
from scipy.integrate import quad
import numpy as np

# FEE CLASS
class BaseFee(ABC):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def calculate_fee(self, transaction_dict: dict, fee_asset: str, **kwargs) -> dict:
        raise NotImplementedError
    
# BASIC PERCENT FEE
class PercentFee(BaseFee):
    def __init__(self, percent: float) -> None:
        super().__init__()
        assert 0. <= percent <= 1.
        self.fee_percent = percent
    def calculate_fee(self, transaction_dict: Dict[str, float], fee_asset: str, **kwargs) -> dict:
        assert fee_asset in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        fee_dict = {} # check asset in transaction & define fee storage
        if fee_asset != "L": # dont fee on liquidity transactions
            fee_delta = abs(transaction_dict[fee_asset]) * self.fee_percent # apply fixed percetn fee
            fee_dict[fee_asset] = fee_delta # update fees dict - adds to existing amount, maybe for multi-step-fee transactions
        return fee_dict # return final fee dict

# NO FEE
class NoFee(PercentFee):
    def __init__(self) -> None:
        super().__init__(0.0)

# TRIANGLE FEE 
class TriangleFee(BaseFee):
    def __init__(self, base_fee: float, min_fee: float, fee_slope: float) -> None:
        """
        calculates fee according to relative change in asset in value due to transaction in AMM
        require: base_fee < min_fee
        base_fee (float): base fee for triangle calc
        min_fee (float): minimum fee for all final triangle fee calcs
        fee_slope (float): slope of fee calc (affects impact of triangle component)(e.g. -1 for charging higher for negative impacts to AMM's asset-in value)
        """
        super().__init__()
        self.base_fee = base_fee
        self.min_fee = min_fee
        self.fee_slope = fee_slope

    def calculate_fee(self, transaction_dict: Dict[str, float], fee_asset: str, **kwargs) -> dict:
        """
        calculates fee according to relative change in asset in value due to transaction in AMM
        NOTE: base_fee < min_fee & selecting allows for adjusting point at which triangle fees start to be applied
        base_fee (float): base fee for value change calc
        min_fee (float): minimum fee for all final triangle fee calcs
        fee_slope (float): slope of fee calc (affects impact of triangle component)(e.g. -1 for charging higher for negative impacts to AMM's asset-in value)
        """
        # ensure portfolio is defined
        amm = kwargs.get("amm")
        assert amm is not None, "AMM must be defined for triangle fee."
        asset_out, asset_in = list(transaction_dict.keys())[0], list(transaction_dict.keys())[1] # get str for asset out and asset in
        # ensure fee asset is in transaction
        assert asset_in in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        fee_dict = {}
        # if not liquidity transaction
        if asset_in != "L":
            # get delta x to set upper limit - get asset out amount and transaction info for fee calc
            asset_out_n, info = amm._quote_no_fee(asset_out, asset_in, transaction_dict[asset_in])
            # asset out = delta x, asset in = delta y
            delta_x = info['asset_delta'][asset_out]  # amout of asset out
            delta_y = info['asset_delta'][asset_in]  # amount of asset in
            # get pre-transaction AMM inventory (X,Y) to calc change in asset-in value due to transaction - define additional params
            base_fee, min_fee, fee_slope, X, Y = self.base_fee, self.min_fee, self.fee_slope, amm.portfolio[asset_out], amm.portfolio[asset_in]
            # calc change in AMM value of asset-in due to transaction
            delta_val = ((Y + delta_y) / (X - delta_x)) - (Y / X)
            # calc final fee adding to base fee for transactions that are of same sign as slope (e.g. slope of
            # -1 and delta_val of asset-in is negative, fee will be addition of their product, base + n)
            end_fee = max([(min_fee), (base_fee + (fee_slope * delta_val))])
            # update fee dict with fee asset and final fee
            fee_dict[fee_asset] = end_fee
        # return fee info with calculated fee
        return fee_dict



class ILossFee(BaseFee):
    def __init__(self, base_fee: float, min_fee: float, fee_slope: float) -> None:
        super().__init__()
        self.base_fee = base_fee
        self.min_fee = min_fee
        self.fee_slope = fee_slope
# # TODO: FINISH B4 IMPLEMENTING
    def calculate_fee(self, transaction_dict: Dict[str, float], fee_asset: str, **kwargs) -> dict:
        # ensure portfolio is defined
        amm = kwargs.get("amm")
        assert amm is not None, "AMM must be defined for triangle fee."
        asset_out, asset_in = list(transaction_dict.keys())[0], list(transaction_dict.keys())[1] # get str for asset out and asset in
        # ensure fee asset is in transaction
        assert asset_in in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        # find change in AMM market value
        # mvalue_t0 = amm. # # TODO: HAVE HISTORY BE PART OF AMM SO CAN FEE BASED ON MARKET VALUE
        fee_dict = {}
        if asset_in != "L":
            asset_out_n, info = amm._quote_no_fee(asset_out, asset_in, transaction_dict[asset_in]) # get delta x to set upper limit
            
        return 0


    # TODO: make market value fee calc - need to add marketDF to AMM



# OLD FEE CALCULATION METHODS FOR TRIANGLE FEE
        # OLD FEE FINAL CALCULATION
        # drew: why is fee averaged? i think changing slope will allow us to parametrically affect the impact of triangle part
        #       also, dont we want min_fee to be actual minimum fee allowed? i thought base_fee was more of that average component we wanted
        
        # if end_fee != min_fee: # check if fee is not min fee
        #         fee_dict[fee_asset] = ((base_fee + end_fee) / 2) # return average of the base and end fees if not min fee
        #         return fee_dict # return fee info
        #     excess_change = (min_fee - (base_fee + (fee_slope * delta_val))) # calculate excess change if fee is min fee
            
        #     total = base_fee - min_fee + excess_change # calculate total fee basis based on base fee less min fee plus excess change
            
        #     fee_dict[fee_asset] =  ((((base_fee + min_fee) / 2) * ((base_fee - min_fee) / total)) + (min_fee * (excess_change / total))) # calculate final fee based on weighted proportions of base and min fees based on total change

        # OLD INTEGRAND METHOD
        # # get delta x to set upper limit
        #     asset_out_amt, info = amm.quote(
        #         asset_out, asset_in, transaction_dict[asset_in])
        #     # define triangle integrand

        #     def _tri_integrand(w, x, y, f, m):
        #         return f + m * (((x * y) / (x - w)**2) - (y / x))
        #     # get fee
        #     # function, lower bound, upper bound, args
        #     fee, error = quad(_tri_integrand, 0, abs(asset_out_amt), args=(
        #         amm.portfolio[asset_out], amm.portfolio[asset_in],
        #         self.max_fee, self.fee_slope))
        #     # save fee
        #     fee_dict[asset_out] = fee