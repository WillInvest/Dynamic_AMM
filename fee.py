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

    def calculate_fee(self, transaction_dict: Dict[str, float], fee_asset: str, **kwargs) -> dict:
        # ensure portfolio is defined
        amm = kwargs.get("amm")
        assert amm is not None, "AMM must be defined for triangle fee."
        # ensure fee asset is in transaction
        assert fee_asset in transaction_dict, f"Fee asset has to be one of the assets in the transaction."
        fee_dict = {}
        if fee_asset != "L":
            # use transaction dict to get asset quantities
            transact_keys = list(transaction_dict.keys())
            receive_asset = [
                asset for asset in transact_keys if asset != fee_asset][0]
            # get delta x to set upper limit
            asset_out, info = amm._quote_no_fee(
                receive_asset, fee_asset, transaction_dict[fee_asset])
            # define integrand
            print("receive", receive_asset, "out", asset_out)
            print("info", info)

            def _tri_integrand(w, x, y, f, m):
                return f + m * (((x * y) / (x - w)**2) - (y / x))
            # get fee
            fee, error = quad(_tri_integrand, 0, asset_out, args=(
                amm.portfolio[receive_asset], amm.portfolio[fee_asset], self.max_fee, self.fee_slope))
            print("FEE:", fee)
            # charge fee based on shift
            # # i think getting fee from dict is unnecessary but for sake of consistency
            fee_dict[fee_asset] = fee  # fee_dict.get(fee_asset, 0.) + fee
        return fee_dict

    # def calculate_fee(self, transaction_dict: dict, fee_asset: str, **kwargs) -> dict:

    #     fee_dict = {}
    #     if fee_asset != "L": # TODO: logic for "L"
    #         # These values can be changed, simply here to test if it works
    #         # Also implementation can be easily changed, again just testing
    #         # bracket_fees = {0.1: (100, 0.01), 0.2: (200, 0.02), 0.3: (300, 0.03)}
    #         # keep track of change coming in
    #         tracker = float(transaction_dict[fee_asset]) # TODO for nagative transaction_dict[fee_asset]
    #         try:
    #             while tracker > 0.0:
    #                 for amount, fee in self.bracket_fees.items():
    #                     fee_delta = min(amount, tracker) * fee
    #                     # Charge fee based on size/remaining size of order
    #                     fee_dict[fee_asset] = fee_dict.get(fee_asset, 0.) +  fee_delta
    #                     # Update fee assets
    #                     # print("HERE1:",
    #                     #       fee_delta, fee_assets[asset])
    #                     # self.fees[fee_asset] += fee_delta
    #                     # print("HERE2", fee_assets[asset])
    #                     tracker -= amount  # Reduce delta remaining by how much we have already assessed fees
    #                     if tracker < 1e-10:  # You can adjust the epsilon value as needed
    #                         break
    #         # catch any errors - added bcs not sure if theres edge case whole in the above logic
    #         except Exception as e:
    #             print(f"Assertion Error: {e}")

        # return updated changes dicitonary
        # print(transaction_dict[fee_asset])
        # return fee_dict
