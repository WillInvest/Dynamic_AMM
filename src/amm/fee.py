# importing src directory
import sys
sys.path.append('..')
# library imports
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
        super().__init__()
        self.base_fee = base_fee
        self.min_fee = min_fee
        self.fee_slope = fee_slope

    def calculate_fee(self, transaction_dict: Dict[str, float], fee_asset: str, **kwargs) -> dict:
        # ensure portfolio is defined
        amm = kwargs.get("amm")
        assert amm is not None, "AMM must be defined for triangle fee."
        asset_out, asset_in = list(transaction_dict.keys())[0], list(transaction_dict.keys())[1] # get str for asset out and asset in
        # ensure fee asset is in transaction
        assert asset_in in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        fee_dict = {}
        if asset_in != "L":
            asset_out_n, info = amm._quote_no_fee(asset_out, asset_in, transaction_dict[asset_in]) # get delta x to set upper limit
            delta_x = amm.portfolio[asset_out] - info['asset_delta'][asset_out] # change in asset out inventory
            delta_y = info['asset_delta'][asset_in] - amm.portfolio[asset_in] # change in asset in inventory
            base_fee, min_fee, slope, X, Y = 0.003, 0.0001, -1, amm.portfolio[asset_out], amm.portfolio[asset_in]
            # ASK SEAN TO COMMENT THIS
            delta_val = ((Y + delta_y) / (X - max(base_fee, delta_x))) - (Y / X) # change in value of asset-in
            end_fee = max([(min_fee), (base_fee + (slope * delta_val))]) # calculate fee
            if end_fee != min_fee: 
                fee_dict[fee_asset] = ((base_fee + end_fee) / 2)   # return fee if not min fee
                return fee_dict # return fee info
            excess_change = (min_fee - (base_fee + (slope * delta_val)))
            total = base_fee - min_fee + excess_change
            fee_dict[fee_asset] =  ((((base_fee + min_fee) / 2) * ((base_fee - min_fee) / total)) + (min_fee * (excess_change / total)))
        return fee_dict # return fee info


