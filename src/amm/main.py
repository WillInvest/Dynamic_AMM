# utility libraries
from utils import parse_input
import pandas as pd
# project libraries
from amm import AMM, SimpleFeeAMM
from fee import TriangleFee, PercentFee, NoFee



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
        if succ: print(f"User Receives: {abs(info['asset_delta'][asset_out])}{asset_out}")
        else: print(f"unsuccessful trade: {info}") # print completion message
        print("Updated portfolio:")
        print(amm) # print updated amm
        print() # print blank line for user prompt

def experiment1():
    amm_sims = [] # create list to store dfs from each simulation of amms
    for set in range(1000): # create new amms & run price path on each N times
        nofeeAMM = SimpleFeeAMM(fee_structure = NoFee()) # create new different AMM types
        percentAMM = SimpleFeeAMM(fee_structure = PercentFee(0.01))
        triAMM = SimpleFeeAMM(fee_structure = TriangleFee(0.003, 0.0001, -1)) # setup dfs to store simulations
        nofeeDF = pd.DataFrame(colummns=["AInv", "BInv", "LInv", "A", "B", "L", "FA", "FB", "FL"])
        percentDF = pd.DataFrame(colummns=["AInv", "BInv", "LInv", "A", "B", "L", "FA", "FB", "FL"])
        triDF = pd.DataFrame(colummns=["AInv", "BInv", "LInv", "A", "B", "L", "FA", "FB", "FL"])
        amms = [(nofeeAMM, nofeeDF), (percentAMM, percentDF), (triAMM, triDF)] # store pairs for updating
        for trade in range(10000): # run trades for each set of AMMs
        # # ADD TRADE CALLS FROM AGENTS HERE # #
            asset_out, asset_in, asset_in_n = "A", "B", 1 # parse input
        # # ADD TRADE CALLS FROM AGENTS HERE # #
            succ1, info1 = nofeeAMM.trade_swap(asset_out, asset_in, asset_in_n) # call trade for each AMM
            succ2, info2 = triAMM.trade_swap(asset_out, asset_in, asset_in_n) # (for each fee type)
            succ3, info3 = percentAMM.trade_swap(asset_out, asset_in, asset_in_n)
            for amm, df in amms: # update dfs with each trade
                new_row = {'AInv': amm.portfolio['A'], 'BInv': amm.portfolio['B'], 'LInv': amm.portfolio['L'], 
                        'A': info1['asset_delta']['A'], 'B': info1['asset_delta']['B'], 'L': info1['asset_delta']['L'], 
                        'FA': amm.fees['A'], 'FB': amm.fees['B'], 'FL': amm.fees['L']}
                df = df.append(new_row, ignore_index=True)

        

if __name__ == "__main__":
    # main()
    experiment1()






