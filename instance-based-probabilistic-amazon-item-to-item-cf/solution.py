import math
from collections import defaultdict

def recommend(
    num_users: int,
    num_items: int,
    purchases: list[tuple[int, int]],
    cart: list[int],
    k: int,
) -> list[tuple[int, float]]:
    """
    "Customers who bought this also bought...": the top k items to recommend
    for a shopping cart, by item-to-item collaborative filtering.
    """
    # 1. Deduplicate cart and pre-calculate lengths/sets for items
    cart_set = set(cart)
    
    # Track the set of users for each item (P_x)
    item_users = defaultdict(set)
    for u, i in purchases:
        item_users[i].add(u)
        
    # Precompute square roots of lengths for items with users to speed up similarity division
    # sqrt_lengths[x] = sqrt(|P_x|)
    sqrt_lengths = {i: math.sqrt(len(users)) for i, users in item_users.items()}
    
    # Map each user to the candidate items they bought
    user_items = defaultdict(list)
    for i, users in item_users.items():
        for u in users:
            user_items[u].append(i)

    # 2. Accumulate scores only for items that share users with cart items
    # scores[j] = sum_{c in cart} sim(c, j)
    scores = defaultdict(float)
    
    for c in cart_set:
        if c not in item_users or not item_users[c]:
            continue
            
        c_users = item_users[c]
        sqrt_len_c = sqrt_lengths[c]
        
        # Count overlaps |P_c & P_j| by iterating over users who bought c
        overlap_counts = defaultdict(int)
        for u in c_users:
            for j in user_items[u]:
                if j not in cart_set:
                    overlap_counts[j] += 1
                    
        # Compute and add similarity for each candidate item j
        for j, overlap in overlap_counts.items():
            sim = overlap / (sqrt_len_c * sqrt_lengths[j])
            scores[j] += sim

    # 3. Filter out zero scores, sort by rules, and return top k
    # Tie-breaking logic: round to 9 decimal places for score equality, 
    # highest score first, then smaller item id first (negated id if sorting descending)
    candidates = []
    for item_id, score in scores.items():
        if score > 0:
            candidates.append((item_id, score))
            
    # Sort criteria: -round(score, 9) sorts highest scores first.
    # item_id sorts smaller item ids first when scores tie.
    candidates.sort(key=lambda x: (-round(x[1], 9), x[0]))
    
    return candidates[:k]
