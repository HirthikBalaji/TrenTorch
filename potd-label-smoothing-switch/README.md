# THE LABEL-SMOOTHING SWITCH

Intermediate | deep-learning | cross-entropy

**Difficulty:** Medium
**Tags:** Deep Learning, Cross-Entropy, Regularization

---

### Story

A classification run enables label smoothing, but the loss implementation
smooths only the target class instead of distributing probability mass
across every class. Compute the correctly smoothed target distribution and
its cross-entropy loss.

---

### The Math

For `C` classes, true class `y`, and smoothing `eps` in `[0, 1]`:

```
q[c] = 1 - eps + eps/C   if c == y
q[c] = eps/C             otherwise
```

Given logits `z`, compute `L = -sum_c q[c] * log_softmax(z)[c]`.

### Input Format

`logits` (length `C`), `y` (0-based), `eps`.

### Output Format

`(q, loss)`: smoothed target probabilities and the loss, 6 decimals.

### Constraints

- `2 <= C <= 10^5`, `|z_c| <= 10^3`, `eps` in `[0, 1]`.
- Time limit: 1.0 second.

---

### Example

**Input**

```
logits = [2, 0, 1], y = 1, eps = 0.1
```

**Output**

```
q = [0.033333, 0.933333, 0.033333]
loss = 2.307606
```

**Explanation:** The true class gets `1 - 0.1 + 0.1/3`, every other class
gets `0.1/3`. A stable log-softmax then gives the loss above.
