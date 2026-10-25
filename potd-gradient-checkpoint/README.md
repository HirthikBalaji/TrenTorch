# THE GRADIENT CHECKPOINT

Advanced | autograd | neural-networks

**Difficulty:** Hard
**Tags:** Autograd, Chain Rule, Neural Networks

---

### Story

A custom neural-network operator passes forward tests but fails gradient checks.
For a tiny two-layer scalar network, work out the exact derivative of squared
error with respect to the first weight, accounting for both activation
derivatives.

---

### The Math

```
h = ReLU(w1*x + b1)
yhat = w2*h + b2
L = 0.5 * (yhat - y)^2
```

`ReLU'(z) = 1` if `z > 0`, otherwise `0`. Compute `dL/dw1` by the chain rule.

### Input Format

`x, y, w1, b1, w2, b2` as function arguments.

### Output Format

`(h, yhat, loss, dL_dw1)`, each a float.

### Constraints

- All magnitudes at most `10^4`.
- Time limit: 1.0 second.

---

### Example

**Input**

```
x=2, y=5, w1=1, b1=0, w2=3, b2=1
```

**Output**

```
h=2.000000 yhat=7.000000 loss=2.000000 dL_dw1=12.000000
```

**Explanation:** `yhat = 7`, so `dL/dyhat = 2`. Then `dyhat/dh = 3` and
`dh/dw1 = x = 2`, giving gradient `2 * 3 * 2 = 12`.
