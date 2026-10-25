import numpy as np

def label_smoothing(logits: np.ndarray, y: int, eps: float) -> tuple[np.ndarray, float]:
    """
    Build the label-smoothed target distribution over C = len(logits) classes 
    and compute its cross-entropy loss against logits.
    
    q[c] = 1 - eps + eps/C for c == y, else eps/C.
    loss = -sum(q * log_softmax(logits)).
    Return (q, loss).
    """
    C = len(logits)
    
    # 1. Initialize every class with the base smoothing mass eps/C
    q = np.full(C, eps / C)
    
    # 2. Add the remaining target mass (1 - eps) onto the true class y
    q[y] += 1.0 - eps
    
    # 3. Compute log_softmax safely using the log-sum-exp trick for numerical stability
    shifted_logits = logits - np.max(logits)
    log_softmax = shifted_logits - np.log(np.sum(np.exp(shifted_logits)))
    
    # 4. Compute cross-entropy loss
    loss = -np.sum(q * log_softmax)
    
    return q, float(loss)
