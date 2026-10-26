def gradient_accumulator(
    theta: float, eta: float, microbatches: list[tuple[int, float]]
) -> tuple[float, float]:
    """
    microbatches: list of (n_j, g_j) pairs, n_j examples with mean gradient g_j.
    g = sum(n_j * g_j) / sum(n_j) (example-weighted mean, NOT the mean of the microbatch means).
    theta' = theta - eta * g.
    Return (g, theta_prime).
    """
    total_examples = 0
    weighted_gradient_sum = 0.0

    for n_j, g_j in microbatches:
        total_examples += n_j
        weighted_gradient_sum += n_j * g_j

    g = weighted_gradient_sum / total_examples
    theta_prime = theta - eta * g

    return g, theta_prime
