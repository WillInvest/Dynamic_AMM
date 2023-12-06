from amm import AMM, SimpleFeeAMM
from utils import parse_input
from fee import TriangleFee, PercentFee


def main():
    # initialize AMM w/ set fee structure
    fee = TriangleFee(0.2, -1)  # PercentFee(0.01)
    amm = SimpleFeeAMM(fee_structure=fee)

    # print initial AMM
    print("Initial AMM: ")
    print(amm)

    # receive trade input
    # while True:
    for i in range(100000):
        print()
        s2string = input("Input string 'Out In In#' (i.e. A B 1): ")
        # check for amm reset
        if s2string == 'r':
            amm = AMM()
            print("Reset amm")
            print(amm)
            continue  # reset

        # parse input
        order = parse_input(s2string)
        asset_out, asset_in, asset_in_amt = order

        # call swap function
        succ, info = amm.trade_swap(asset_out, asset_in, asset_in_amt)

        # print trade info
        if succ:
            print("--------------------")
            print(f"Successful Swap {asset_in} for {asset_out}")
            print(
                f"{asset_in_amt} {asset_in} ---> {abs(info['asset_delta'][asset_out]+info['fee'][asset_out])} {asset_out}")
            print(f"Fee charge {info['fee'][asset_out]} {asset_out}")
            print("--------------------")
        #     print(f"User pays {asset_in_amt} {asset_in}")
        #     print(f"Swap value: {abs(info['asset_delta'][asset_out])}")
        #     print(
        #         f"User gets {abs(info['asset_delta'][asset_out]+info['fee'][asset_out])}{asset_out}")
        #     print(f"User fee {info['fee'][asset_out]}{asset_out}")
        else:
            print(f"Unsuccessful trade: {info}")
        # print updated AMM
        print("Updated portfolio:")
        print(amm)
        print()


if __name__ == "__main__":
    main()
