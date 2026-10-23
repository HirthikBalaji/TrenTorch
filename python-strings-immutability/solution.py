def replace_char_at(s: str, index: int, ch: str) -> str:
    """
    Return a new string equal to `s` with the character at position `index` replaced by `ch`.
    `index` may be negative.
    `ch` may be any string, not necessarily one character long.
    If `index` is out of range (index >= len(s) or index < -len(s)), return `s` unchanged.
    Do not use the replace() method.
    Example: replace_char_at("hello", 0, "J") -> "Jello"
    """
    length = len(s)
    
    # Check boundaries for both positive and negative indices
    if index >= length or index < -length:
        return s
        
    # Convert negative index to its positive equivalent
    if index < 0:
        index += length
        
    # Reconstruct the string using slicing
    return s[:index] + ch + s[index + 1:]


def insert_at(s: str, index: int, text: str) -> str:
    """
    Return a new string with `text` inserted before position `index`.
    Slice semantics apply: an `index` at or beyond len(s) appends, and a negative `index` counts from the end (clamped at the start of the string).
    Examples:
    insert_at("abc", 1, "X") -> "aXbc"
    insert_at("abc", -1, "X") -> "abXc"
    insert_at("abc", 99, "X") -> "abcX"
    """
    # Python slices naturally handle out-of-bounds indices and negative clamping
    return s[:index] + text + s[index:]


def join_with_separator(parts, sep: str) -> str:
    """
    `parts` is a list of strings. Build and return one string containing every element of `parts`,
    with `sep` placed between consecutive elements (not before the first, not after the last).
    Use a for loop with + / += only. Do not use the join() method.
    An empty `parts` returns "".
    Example: join_with_separator(["a", "b", "c"], "-") -> "a-b-c"
    """
    if not parts:
        return ""
        
    # Start with the first element so the separator is only added between elements
    result = parts[0]
    
    for item in parts[1:]:
        result += sep + item
        
    return result


def repeat_text(s: str, n: int) -> str:
    """
    Return `s` repeated `n` times using the * operator.
    If n <= 0, return "".
    """
    if n <= 0:
        return ""
    return s * n


def total_chars_copied(n: int, piece_length: int) -> int:
    """
    Model of repeated concatenation. A string starts empty.
    For each of `n` steps, one piece of length `piece_length` is appended,
    and the step writes every character of the NEW string (previous content plus the piece).
    Return the total number of characters written across all n steps:
    piece_length * (1 + 2 + ... + n).
    If n <= 0, return 0.
    Example: total_chars_copied(3, 2) -> 2 + 4 + 6 = 12
    """
    if n <= 0:
        return 0
        
    # Uses the arithmetic series formula: 1 + 2 + ... + n = n * (n + 1) // 2
    sum_of_n = (n * (n + 1)) // 2
    return piece_length * sum_of_n
