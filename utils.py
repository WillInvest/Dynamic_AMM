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
