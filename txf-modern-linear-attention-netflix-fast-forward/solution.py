import numpy as np


def feature_map(x: np.ndarray) -> np.ndarray:
    """
    Element-wise positive feature map:
        phi(x) = max(x, 0) + 1
    """
    return np.maximum(x, 0) + 1


def linear_attention_row_sums(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> np.ndarray:
    """
    Linear attention without materializing an N x N matrix.

    Q, K, V: arrays of shape (N, d).

    Returns the sum of each row of the attention output, shape (N,).
    Memory complexity: O(N * d).
    """
    # Apply positive feature map
    Q_prime = feature_map(Q)
    K_prime = feature_map(K)

    # Compute M = K'^T @ V, shape (d, d)
    M = K_prime.T @ V

    # Compute Z = sum of K' across all rows, shape (d,)
    Z = np.sum(K_prime, axis=0)

    # Compute numerator: each row of Q' multiplied by M, shape (N, d)
    numerator = Q_prime @ M

    # Compute denominator for each query, shape (N,)
    denominator = Q_prime @ Z

    # Normalize each row and sum its elements
    output_row_sums = np.sum(
        numerator / denominator[:, np.newaxis], axis=1
    )

    return output_row_sums
