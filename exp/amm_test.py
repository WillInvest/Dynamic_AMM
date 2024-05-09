import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.amm.amm import SimpleFeeAMM
from src.amm.fee import PercentFee


def print_trade_results(amm, success, info):
    print(f"Success: {success}")
    print("Trade Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("Updated Portfolio:")
    print(amm)


def main():
    # Test cases without fee
    print("Testing without fee...")
    trades = [
        ("A", "B", 100),
        ("A", "B", -100),
        ("B", "A", 100),
        ("B", "A", -100)
    ]

    for trade in trades:
        # Initialize AMM with no fee
        amm_no_fee = SimpleFeeAMM(
            utility_func="constant_product",
            init_portfolio={'A': 1000, 'B': 1000, 'L': 1000},
            fee_structure=PercentFee(0.0)
        )
        asset_out, asset_in, amount = trade
        print(f"\nTrade: {asset_in} -> {asset_out} Amount: {amount}")
        success, info = amm_no_fee.trade_swap(asset_out, asset_in, amount)
        print_trade_results(amm_no_fee, success, info)

    # Deplete the pool test without fee
    print("\nAttempting to deplete the pool...")
    deplete_trades = [
        ("A", "B", 2000),
        ("A", "B", -2000),
        ("B", "A", 2000),
        ("B", "A", -2000)
    ]

    for trade in deplete_trades:
        # Initialize AMM with no fee
        amm_no_fee = SimpleFeeAMM(
            utility_func="constant_product",
            init_portfolio={'A': 1000, 'B': 1000, 'L': 1000},
            fee_structure=PercentFee(0.0)
        )
        asset_out, asset_in, amount = trade
        print(f"\nTrade: {asset_in} -> {asset_out} Amount: {amount}")
        try:
            success, info = amm_no_fee.trade_swap(asset_out, asset_in, amount)
            print_trade_results(amm_no_fee, success, info)
        except AssertionError as e:
            print(f"AssertionError: {e}")

    # Test cases with fee
    print("\nTesting with fee...")
    trades_with_fee = [
        ("A", "B", 100),
        ("A", "B", -100),
        ("B", "A", 100),
        ("B", "A", -100)
    ]

    for trade in trades_with_fee:
        # Initialize AMM with a fee
        amm_with_fee = SimpleFeeAMM(
            utility_func="constant_product",
            init_portfolio={'A': 1000, 'B': 1000, 'L': 1000},
            fee_structure=PercentFee(0.01)  # 1% fee
        )
        asset_out, asset_in, amount = trade
        if amount >= 0:
            print(f"\nTrade: {asset_in} -> {asset_out} Amount: {amount}")
        else:
            print(f"\nTrade: {asset_out} -> {asset_in} Amount: {amount}")
        success, info = amm_with_fee.trade_swap(asset_out, asset_in, amount)
        print_trade_results(amm_with_fee, success, info)

    # Deplete the pool test with fee
    print("\nAttempting to deplete the pool with fee...")
    for trade in deplete_trades:
        # Initialize AMM with a fee
        amm_with_fee = SimpleFeeAMM(
            utility_func="constant_product",
            init_portfolio={'A': 1000, 'B': 1000, 'L': 1000},
            fee_structure=PercentFee(0.01)  # 1% fee
        )
        asset_out, asset_in, amount = trade
        if amount >= 0:
            print(f"\nTrade: {asset_in} -> {asset_out} Amount: {amount}")
        else:
            print(f"\nTrade: {asset_out} -> {asset_in} Amount: {amount}")
        try:
            success, info = amm_with_fee.trade_swap(asset_out, asset_in, amount)
            print_trade_results(amm_with_fee, success, info)
        except AssertionError as e:
            print(f"AssertionError: {e}")


if __name__ == '__main__':
    main()
