import math
from typing import List

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Computes standard Euclidean (L2) distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def dot_product(v1: List[float], v2: List[float]) -> float:
    """Computes Inner Product (dot product)."""
    return sum(a * b for a, b in zip(v1, v2))

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes Cosine Similarity (-1.0 to 1.0)."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def cosine_distance(v1: List[float], v2: List[float]) -> float:
    """Computes Cosine Angular Distance (0.0 to 2.0)."""
    return 1.0 - cosine_similarity(v1, v2)
