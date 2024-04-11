# importing src directory
import sys
sys.path.append('..')
# library imports
from typing import Dict
import math

# parse main input (i.e. A B 1))
def parse_input(string: str):
    """parse user input (i.e. A B 1) to tuple of strings and float"""
    inputs = string.split(" ")
    inputs[-1] = float(inputs[-1]) # convert last to #
    return tuple(inputs)


# define fee dictionary class for amm's
class FeeDict(dict):
    # check if fee dict is empty
    def is_empty(self):
        """check if fee dict is empty"""
        return sum(self.values()) == 0
    # reset fees to zero
    def reset(self):
        """reset fee inventory to zero"""
        self.last_fees = dict(self.copy())
        for key in self:
            self[key] = 0.
            







# __TODO__ UPDATE

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

    
# __TODO__ UPDATE
            
def distribute_fees(lp_tokens: dict, fees: FeeDict) -> Dict[str, Dict[str, float]]:
    ret = {key: {sub_key: 0. for sub_key in fees} for key in lp_tokens}
    
    sum_tokens = sum(lp_tokens.values())
    for lp_user in lp_tokens:
        for asset in fees:
            ret[lp_user][asset] += (lp_tokens[lp_user]/sum_tokens)*fees[asset]
            
    return ret

# __TODO__ UPDATE

def add_lp_tokens(lp_tokens: dict, num_tokens: float) -> None:
    assert num_tokens >= 0, f"Added number of tokens should be non-negative: {num_tokens}"
    sum_tokens = sum(lp_tokens.values())
    for lp_user in lp_tokens:
        lp_tokens[lp_user] += (lp_tokens[lp_user]/sum_tokens)*num_tokens
            
# __TODO__ UPDATE ?

def set_market_trade(amm, MP: float, inv1: str, inv2: str) -> None:
    inventory_1 = amm.portfolio[inv1]
    
    inventory_2 = amm.portfolio[inv2]
    
    ratio = inventory_1 / inventory_2

    if ratio > MP:
        y = math.sqrt(inventory_1 * inventory_2/MP) - inventory_2
        amm.trade_swap(inv1,inv2,y)
        #print(f"This is your trade to execute: {inv2} {inv1} {y}")
        
    elif ratio < MP:
        x = math.sqrt(MP * inventory_1 *inventory_2) - inventory_1
        
        amm.trade_swap(inv2,inv1,x)
        #print(f"This is your trade to execute: {inv1} {inv2} {x}")

        