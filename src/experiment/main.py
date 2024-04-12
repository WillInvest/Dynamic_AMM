# importing src directory
import sys
sys.path.append('..')
# utility libraries
from amm.utils import parse_input
# project libraries
from amm.amm import AMM, SimpleFeeAMM
from amm.fee import TriangleFee, PercentFee, NoFee



def main():
    amm = SimpleFeeAMM(fee_structure = TriangleFee(0.003, 0.0001, -1)) # PercentFee(0.01))
    print("Initial AMM: ")
    print(amm) # print initial amm
    for i in range(100000): # run console for trade inputs
        print() # print blank line for user prompt
        inputstr = input("Input string (i.e. A B 1): ")
        if inputstr == 'r':
            amm = AMM()
            print("Reset amm")
            print(amm)
            continue  # reset
        order = parse_input(inputstr) 
        asset_out, asset_in, asset_in_n = order # parse input
        succ, info = amm.trade_swap(asset_out, asset_in, asset_in_n) # get success bool and trade info
        print("INFO")
        print(info)
        print("AMM")
        print(amm.portfolio)
        if succ: # print trade completion message
            print("--------------------")
            print(f"Successful Swap  |  Paid: {abs(info['asset_delta'][asset_in])}{asset_in}  |  Received: {abs(info['asset_delta'][asset_out])}{asset_out}  |  Fee: {info['fee'][asset_out]}{asset_out}")
            print("--------------------")
        else: print(f"Error: unsuccessful trade: {info}") # print completion message
        print("Updated portfolio:")
        print(amm) # print updated amm
        print() # print blank line for user prompt


if __name__ == "__main__":
    main()






