from typing import Dict

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
        
        