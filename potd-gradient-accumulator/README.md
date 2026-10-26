# THE GRADIENT ACCUMULATOR

Advanced | training-systems | optimization

**Difficulty:** Hard
**Tags:** Training Systems, Gradient Accumulation, Optimization

---

### Story

A GPU job simulates a large batch by accumulating gradients from
microbatches. A bug divides by the number of microbatches and then again by
the total example count, making updates far too small. Compute the exact
update under the stated mean-loss convention.

---

### The Math

Microbatch `j` has `n_j` examples and mean gradient `g_j`. The full-batch
mean gradient is the example-count-weighted mean:

```
g = sum(n_j * g_j) / sum(n_j)
```

One SGD update: `theta' = theta - eta * g`. Microbatch sizes may differ;
do not average the microbatch means equally unless the sizes happen to be
equal.

### Input Format

`theta`, `eta`, `microbatches`: a list of `(n_j, g_j)` pairs.

### Output Format

`(g, theta_prime)`, 6 decimal places.

### Constraints

- `1 <= k <= 10^5` microbatches, `1 <= n_j <= 10^6`.
- `|g_j|, |theta| <= 10^4`, `0 < eta <= 1`.
- Time limit: 1.0 second. Memory: 64 MB.

---

### Example

**Input**

```
theta = 10, eta = 0.1, microbatches = [(1, 2), (3, 4)]
```

**Output**

```
g = 3.500000
theta_prime = 9.650000
```

**Explanation:** The example-weighted gradient is `(1*2 + 3*4)/4 = 3.5`, so
`theta' = 10 - 0.1*3.5 = 9.65`.
