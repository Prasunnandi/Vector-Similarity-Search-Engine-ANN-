import heapq
from typing import List, Tuple, Dict, Any
from ann_engine.metrics import euclidean_distance, cosine_distance

class FlatIndex:
    """Brute force exact k-NN index used as the ground truth oracle for Recall@K evaluation."""
    def __init__(self, metric: str = "euclidean"):
        self.metric = metric
        self.dist_fn = euclidean_distance if metric == "euclidean" else cosine_distance
        self.ids: List[int] = []
        self.vectors: List[List[float]] = []
        self.payloads: List[Dict[str, Any]] = []

    def add(self, vector_id: int, vector: List[float], payload: Dict[str, Any] = None):
        self.ids.append(vector_id)
        self.vectors.append(vector)
        self.payloads.append(payload or {})

    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        scores = []
        for vid, vec, pl in zip(self.ids, self.vectors, self.payloads):
            d = self.dist_fn(query, vec)
            scores.append((d, vid, pl))

        # Sort ascending by distance (top k smallest distances)
        scores.sort(key=lambda x: x[0])
        top_k = scores[:k]
        return [(vid, d, pl) for d, vid, pl in top_k]
