# importing src directory
import sys
sys.path.append('..')
# library imports
from typing import Dict
import math
import time

def parse_input(string: str):
    results = string.split(" ")
    results[-1] = float(results[-1])
    return tuple(results)


def add_dict(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries based on common keys.
    
    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.
        
    Returns:
        dict: Merged dictionary.
    """
    merged_dict = dict1.copy()  # Create a copy of the first dictionary to preserve original data
    for key, value in dict2.items():
        if key in merged_dict:
            # If the key is common, add the values from both dictionaries
            merged_dict[key] += value
        else:
            # If the key is unique, add it to the merged dictionary
            merged_dict[key] = value
    return merged_dict   

        
class FeeDict(dict):

    def is_empty(self):
        return sum(self.values()) == 0
    
    def reset(self):
        self.last_fees = dict(self.copy())
        for key in self:
            self[key] = 0.
            
    
            
            
def distribute_fees(lp_tokens: dict, fees: FeeDict) -> Dict[str, Dict[str, float]]:
    ret = {key: {sub_key: 0. for sub_key in fees} for key in lp_tokens}
    
    sum_tokens = sum(lp_tokens.values())
    for lp_user in lp_tokens:
        for asset in fees:
            ret[lp_user][asset] += (lp_tokens[lp_user]/sum_tokens)*fees[asset]
            
    return ret

def add_lp_tokens(lp_tokens: dict, num_tokens: float) -> None:
    assert num_tokens >= 0, f"Added number of tokens should be non-negative: {num_tokens}"
    sum_tokens = sum(lp_tokens.values())
    for lp_user in lp_tokens:
        lp_tokens[lp_user] += (lp_tokens[lp_user]/sum_tokens)*num_tokens
            
        
def set_acc_market_trade(amm, MP: float, invB: str, invA: str) -> None:
    arbitFolio = {}
    fee_ = "Fee"
    current_B_Price_wfee = 0
    if hasattr(amm.fee_structure, 'fee_percent') and amm.fee_structure.fee_percent== 0.0:
        fee_ = "No Fee"
        current_B_Price_wfee, info = amm._quote_no_fee(invB,invA,1)
    else:
        fee_ = "Fee"
        current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)

    if abs(current_B_Price_wfee) < MP:
        k2 = amm.portfolio[invA] - math.sqrt((amm.portfolio[invB]*amm.portfolio[invA])/MP)
        print(f'K2 value is : {abs(k2)}')
        if k2<0:
            k2 = k2*(-1)
        current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
        print("B's MP before arbitrage trade: ",abs(current_B_Price_wfee))
        amm.fee_precharge = True
        success,temp = amm.trade_swap(invB,invA,-k2)
        amm.fee_precharge = False
        current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
        print("B's MP after arbitrage trade: ",abs(current_B_Price_wfee))
    if abs(current_B_Price_wfee) > MP:
        k2 = amm.portfolio[invB] - math.sqrt(amm.portfolio[invB]*amm.portfolio[invA]*MP)
        print(f'K2 value is : {abs(k2)}')
        if k2<0:
            k2 = k2*(-1)
        current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
        print("B's MP before arbitrage trade: ",abs(current_B_Price_wfee))
        amm.fee_precharge = True
        success,temp = amm.trade_swap(invA,invB,-k2)
        amm.fee_precharge = False
        current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
        print("B's MP after arbitrage trade: ",abs(current_B_Price_wfee))

def set_market_trade(amm, MP: float, invB: str, invA: str) -> None:    
    arbitFolio = {}
    fee_ = "Fee"
    current_B_Price_wfee = 0
    if hasattr(amm.fee_structure, 'fee_percent') and amm.fee_structure.fee_percent== 0.0:
        fee_ = "No Fee"
        current_B_Price_wfee, info = amm._quote_no_fee(invB,invA,1)
    else:
        fee_ = "Fee"
        current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
    # print("B's MP per 1 stock of A is: ",abs(current_B_Price_wfee))

    if abs(current_B_Price_wfee) < MP:
        # sim_order = -0.1
        sim_order = 1
        count=0
        arbitFolio[invB]= 0
        arbitFolio[invA]= 0

        #we are gonna execute small orders until the Price of B to A is returned to the MP
        while abs(current_B_Price_wfee) < MP:
            # amm.trade_swap(invB,invA,sim_order)
            sim_order = 1
            success,temp = amm.trade_swap(invA,invB,sim_order)
            if success:
                temp = temp['pay_s1']
            else:
                temp=0
                sim_order=0
            arbitFolio[invA] -= temp
            arbitFolio[invB] -= sim_order
            if fee_ == "No Fee":
                current_B_Price_wfee, info = amm._quote_no_fee(invB,invA,1)
            else:
                current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
            count+=1
            # print("Inside Price",abs(current_B_Price_wfee))
            # print("INv A",amm.portfolio[invA])
            # print("INv b",amm.portfolio[invB])
        # print("THe number of times the loop was run is:",count)
  
    elif abs(current_B_Price_wfee) > MP:
        # sim_order = -0.1
        sim_order = -1
        gcount=0
        arbitFolio[invB]= 0
        arbitFolio[invA]= 0
        #we are gonna execute small orders until the Price of B to A is returned to the MP
        while abs(current_B_Price_wfee) > MP:
            # amm.trade_swap(invB,invA,sim_order)
            sim_order = -1
            success,temp = amm.trade_swap(invA,invB,sim_order)
            if success:
                temp = temp['pay_s1']
            else:
                temp=0
                sim_order=0
            arbitFolio[invA] -= temp
            arbitFolio[invB] -= sim_order
            if fee_ == "No Fee":
                current_B_Price_wfee, info = amm._quote_no_fee(invB,invA,1)
            else:
                current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
            gcount+=1
        # print("THe number of times the gloop was run is:",gcount)
    #At this point we must have over valued B
    # if fee_ == "No Fee":
    #     current_B_Price_wfee, info = amm._quote_no_fee(invB,invA,1)
    # else:
    #     current_B_Price_wfee, info = amm._quote_post_fee(invB,invA,1)
    # print("B's MP(After thread) per 1 stock of A is: ",abs(current_B_Price_wfee))
    # print("THe arbitrage agent;s portfolio is:",arbitFolio)


    #Just have to add the code so as to see if the market is profitable or not 

        