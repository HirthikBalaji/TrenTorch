# AMAZON ITEM-TO-ITEM CF

Intermediate | classical-ml | recommender-systems | collaborative-filtering | sparse-linear-algebra

**Difficulty:** Medium
**Tags:** Recommendation Systems, Collaborative Filtering, Sparse Linear Algebra, Hash Maps

---

### Story

In 2003, Amazon engineers faced a massive scaling problem. Standard "user-to-user"
collaborative filtering had to compare a customer against millions of other customers in real
time, which is an impossible amount of computation for a page that has to load instantly. They
published a different approach: **item-to-item collaborative filtering**. Instead of finding
similar users, find similar _items_ from historical co-purchase data, so that inference becomes
a fast, local lookup.

You are implementing the core inference engine for the "Customers who bought this also
bought..." widget. Given a customer's active shopping cart, you scan the historical purchase
logs, compute the cosine similarity between each item in the cart and the rest of the catalog,
and return the top recommended items. The catalog is huge but very sparse, so you must not
allocate a dense `I x I` similarity matrix. Compute the scores on demand by walking sparse
adjacency lists.

---

### The ML: Item-to-Item Collaborative Filtering

The catalog has `I` items, and you are given purchase logs from `U` users.

For an item `x`, let `P_x` be the set of users who bought it. The similarity between two items
`x` and `y` is their **cosine similarity**:

```
sim(x, y) = |P_x ∩ P_y| / ( sqrt(|P_x|) * sqrt(|P_y|) )
```

If either `P_x` or `P_y` is empty, `sim(x, y) = 0`.

Given a target customer's cart, a set of items `C`, score every candidate item `j` that is not
in the cart. The score is the sum of its similarities to every item in the cart:

```
Score(j) = sum over c in C of sim(c, j)
```

Return the top `K` items by score. Scores that agree to within `1e-9` count as equal, and a tie
goes to the **smaller item ID**.

---

### Input Format

```
U I E C K
u_1 i_1
u_2 i_2
...
u_E i_E
c_1
c_2
...
c_C
```

- `U, I, E, C, K` are the number of users, items, purchase events, cart items, and requested
  recommendations. Users and items are 0-indexed.
- The next `E` lines each hold two integers `u i`: user `u` purchased item `i`. A user may
  purchase the same item several times. Repeated purchases of one item by one user count as a
  single member of `P_i`.
- The last `C` lines are the item IDs currently in the cart. The cart is a set: an ID listed
  more than once counts once.

### Output Format

Print up to `K` lines, fewer if fewer than `K` candidate items have a score above `0` (and
nothing at all if none do). Each line holds a recommended item ID and its score, separated by a
space, with the score to exactly 6 decimal places.

Sort by score descending, then by item ID ascending.

**Judging:** accepted if every score is within `1e-5` absolute error of the reference solution
and the item order matches.

---

### Constraints

- `1 <= U <= 10^5`, `1 <= I <= 10^5`
- `0 <= E <= 3 * 10^5`
- `1 <= C <= 100`
- `1 <= K <= 100`
- `0 <= u < U`, `0 <= i < I` for every purchase event, and every cart ID satisfies `0 <= c < I`
- Time limit: 1.0 second. Memory limit: 128 MB. A dense `I x I` matrix at `I = 10^5` needs
  about 80 GB, so building one fails immediately.

---

### Example 1

**Input**

```
4 5 9 2 2
0 0
0 1
0 2
1 0
1 1
2 1
2 3
3 0
3 4
0
1
```

**Output**

```
2 1.154701
3 0.577350
```

**Explanation:** The cart is `{0, 1}`. Reading the purchase events off by item:

```
P_0 = {0, 1, 3}    P_1 = {0, 1, 2}    P_2 = {0}    P_3 = {2}    P_4 = {3}
```

Candidates are items 2, 3 and 4.

- **Item 2** is bought only by user 0, who also bought both cart items. `sim(0, 2) = 1 / (sqrt(3) * 1)`
  and `sim(1, 2) = 1 / (sqrt(3) * 1)`, so `Score(2) = 2 / sqrt(3) = 1.154701`.
- **Item 3** is bought only by user 2. It shares no buyer with item 0, so `sim(0, 3) = 0`. It shares
  user 2 with item 1, so `sim(1, 3) = 1 / sqrt(3)` and `Score(3) = 0.577350`.
- **Item 4** is bought only by user 3. `sim(0, 4) = 1 / sqrt(3)` and `sim(1, 4) = 0`, so
  `Score(4) = 0.577350`.

With `K = 2`, item 2 comes first. Items 3 and 4 tie for second, and item 3 wins on the smaller ID.

---

### Example 2

**Input**

```
3 4 5 1 1
0 0
0 1
1 0
1 2
2 3
0
```

**Output**

```
1 0.707107
```

**Explanation:** The cart is `{0}` and `P_0 = {0, 1}`. Item 1 was bought by `{0}` and item 2 by
`{1}`, so each shares exactly one buyer with item 0:
`1 / (sqrt(2) * sqrt(1)) = 0.707107`. Item 3 shares no buyer, so its score is `0` and it is never
returned. Items 1 and 2 tie, and with `K = 1` item 1 wins on the smaller ID.
