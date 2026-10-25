def gradient_checkpoint(
    x: float, y: float, w1: float, b1: float, w2: float, b2: float
) -> tuple[float, float, float, float]:
    """
    Two-layer scalar network: h = ReLU(w1*x + b1), yhat = w2*h + b2,
    L = 0.5 * (yhat - y)^2.

    Return (h, yhat, loss, dL_dw1).
    """
    # Forward pass
    z = w1 * x + b1
    h = max(0.0, z)
    yhat = w2 * h + b2
    loss = 0.5 * (yhat - y) ** 2

    # Backward pass using the chain rule
    relu_grad = 1.0 if z > 0 else 0.0
    dL_dw1 = (yhat - y) * w2 * relu_grad * x

    return h, yhat, loss, dL_dw1
