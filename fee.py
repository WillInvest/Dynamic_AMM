from abc import ABC, abstractmethod

from typing import Literal, Dict
from scipy.integrate import quad
import numpy as np


class BaseFee(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def calculate_fee(self, transaction_dict: dict, fee_asset: str, **kwargs) -> dict:
        raise NotImplementedError


class PercentFee(BaseFee):
    def __init__(self, percent: float) -> None:
        super().__init__()
        assert 0. <= percent <= 1.
        self.fee_percent = percent

    def calculate_fee(self, transaction_dict: Dict[str, float], fee_asset: str, **kwargs) -> dict:
        assert fee_asset in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        fee_dict = {}
        if fee_asset != "L":
            # Apply fixed percetn fee for buying asset
            fee_delta = abs(transaction_dict[fee_asset]) * self.fee_percent
            # Charge fee based on size of order
            fee_dict[fee_asset] = fee_dict.get(fee_asset, 0.) + fee_delta
            # # Update fee assets
            # self.fees[asset] += fee_delta
        return fee_dict


class NoFee(PercentFee):
    def __init__(self) -> None:
        super().__init__(0.0)


class TriangleFee(BaseFee):
    def __init__(self, max_fee: float, fee_slope: float) -> None:
        super().__init__()
        self.max_fee = max_fee
        self.fee_slope = fee_slope

    def calculate_fee(self, transaction_dict: Dict[str, float], asset_out: str, asset_in: str, **kwargs) -> dict:
        # ensure portfolio is defined
        amm = kwargs.get("amm")
        assert amm is not None, "AMM must be defined for triangle fee."
        # ensure fee asset is in transaction
        assert asset_in in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        fee_dict = {}
        if asset_in != "L":
            # get delta x to set upper limit
            asset_out_amt, info = amm.quote(
                asset_out, asset_in, transaction_dict[asset_in])
            # define triangle integrand

            def _tri_integrand(w, x, y, f, m):
                return f + m * (((x * y) / (x - w)**2) - (y / x))
            # get fee
            # function, lower bound, upper bound, args
            fee, error = quad(_tri_integrand, 0, abs(asset_out_amt), args=(
                amm.portfolio[asset_out], amm.portfolio[asset_in],
                self.max_fee, self.fee_slope))
            # save fee
            fee_dict[asset_out] = fee
        # return fee
        return fee_dict
