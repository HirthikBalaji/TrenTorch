# THE CAUSAL MASK MISFIRE

Advanced | transformers | attention

**Difficulty:** Hard
**Tags:** Transformers, Attention Masks, Tensor Semantics

---

### Story

A transformer serving kernel uses an additive causal mask. It accidentally
adds zero to forbidden logits instead of a large negative value, letting
future tokens leak attention. Given a raw score matrix, compute the
attention probabilities a correctly masked kernel should have produced.

---

### The Math

For an `n x n` score matrix `S`, key position `j` is allowed for query
position `i` iff `j <= i` (zero-based). Define

```
S_tilde[i][j] = S[i][j]   if j <= i
S_tilde[i][j] = -inf      if j > i
```

Apply row-wise, numerically stable softmax to `S_tilde`.

### Input Format

A 2D array `scores` (`n x n`).

### Output Format

A 2D array of probabilities, same shape, 6 decimal places. Forbidden future
positions must be exactly zero.

### Constraints

- `1 <= n <= 300`, `|scores[i][j]| <= 10^4`.
- Time limit: 2.0 seconds. Memory: 64 MB.

---

### Example

**Input**

```
scores = [[1, 2, 3],
          [1, 2, 3],
          [1, 2, 3]]
```

**Output**

```
[[1.000000, 0.000000, 0.000000],
 [0.268941, 0.731059, 0.000000],
 [0.090031, 0.244728, 0.665241]]
```

**Explanation:** Row 0 can attend only to its first key; row 1 to the first
two; row 2 to all three.
