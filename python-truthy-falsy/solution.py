def is_truthy(value) -> bool:
    """ 
    Return the truthiness of `value` exactly as Python's own rules would determine it 
    (equivalent to bool(value), but implement the logic explicitly rather than calling bool() directly, 
    handle None, numbers, strings, lists, tuples, dicts, and sets). 
    """
    # None is always falsy
    if value is None:
        return False
    
    # Check for boolean types directly
    if isinstance(value, bool):
        return value
        
    # Numbers (int, float, complex): 0 is falsy, everything else is truthy
    if isinstance(value, (int, float, complex)):
        return value != 0
        
    # Collections / Sequences (str, list, tuple, dict, set): empty collections are falsy
    if isinstance(value, (str, list, tuple, dict, set)):
        return len(value) > 0
        
    # Fallback default for custom objects or other types
    return True


def first_truthy(values: list):
    """ 
    Return the first value in `values` that is truthy. 
    If no value in the list is truthy, return None. 
    Do not use Python's built-in `any()`, implement the scan explicitly using is_truthy-style checks. 
    """
    for item in values:
        if is_truthy(item):
            return item
            
    return None
