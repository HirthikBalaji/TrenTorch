def run_pipeline(values: list, *transformations, **options) -> list:
    """
    Apply a sequence of transformation functions to every value in `values` 
    and return the resulting values as a new list. Each transformation must 
    accept one value and return one value. Transformations are applied in 
    the order supplied.

    Supported keyword options:
    unique        Boolean. If True, include each final output value only once,
                  preserving the order of its first occurrence. Defaults to False.
    limit         Integer or None. If an integer is supplied, stop after collecting
                  at most `limit` output values. Defaults to None, meaning no limit.
                  `limit` must not be negative.

    Unknown keyword options must raise `TypeError`.
    If `unique=True`, use a set to track values already emitted and a list to 
    preserve the output order.
    If `limit == 0`, return an empty list.
    The input `values` must not be modified.

    Example:
    def double(x): return x * 2
    def increment(x): return x + 1
    run_pipeline([1, 2, 3], double, increment) -> [3, 5, 7]
    run_pipeline([1, 1, 2, 2], unique=True) -> [1, 2]
    """
    # 1. Validate keyword arguments
    allowed_options = {'unique', 'limit'}
    for key in options:
        if key not in allowed_options:
            raise TypeError(f"run_pipeline() got an unexpected keyword argument '{key}'")
            
    # 2. Extract options and apply defaults
    unique = options.get('unique', False)
    limit = options.get('limit', None)
    
    # 3. Validate limit value
    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        if limit == 0:
            return []

    # 4. Initialize helpers for processing
    output = []
    seen = set() if unique else None

    # 5. Process each value through the pipeline
    for item in values:
        # Check if limit has been reached
        if limit is not None and len(output) >= limit:
            break
            
        # Apply all transformation functions sequentially
        transformed_item = item
        for transform in transformations:
            transformed_item = transform(transformed_item)
            
        # Apply uniqueness logic
        if unique:
            if transformed_item not in seen:
                seen.add(transformed_item)
                output.append(transformed_item)
        else:
            output.append(transformed_item)

    # 6. Apply final slicing for limit (in case loop didn't break early)
    if limit is not None:
        return output[:limit]
        
    return output
