def track_calls(func):
    """Decorator that counts function executions using a private attribute."""
    func.calls = 0
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

def make_pipeline(*transformations):
    """
    Return a callable pipeline that applies every supplied one-argument function 
    to its input in the exact order given.
    """
    
    @track_calls
    def pipeline(x):
        # Enclosing scope retains the specific 'transformations' tuple
        current_value = x
        for transform in transformations:
            current_value = transform(current_value)
        return current_value

    return pipeline
