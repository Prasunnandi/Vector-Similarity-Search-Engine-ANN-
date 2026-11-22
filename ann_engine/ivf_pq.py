import math
import random
from typing import List, Tuple, Dict, Any
from ann_engine.metrics import euclidean_distance

class IVFPQIndex:
    """
    Inverted File Index with Product Quantization (IVF-PQ).
    Compresses high-dimensional floating point vectors by 8x to 16x using codebook quantization
    and accelerates search via Voronoi inverted cell indexing.
    """
    def __init__(self, dim: int, nlist: int = 16, m_sub: int = 4, nprobe: int = 4):
        assert dim % m_sub == 0, f"Dimension {dim} must be divisible by sub-vector count {m_sub}"
        self.dim = dim
        self.nlist = nlist          # Number of coarse Voronoi clusters
        self.m_sub = m_sub          # Number of quantized sub-vectors
        self.sub_dim = dim // m_sub
        self.nprobe = nprobe        # Number of adjacent Voronoi cells scanned
        self.k_centroids = 16       # 16 codebook centroids per sub-space (4-bit or 8-bit quantization)

        self.coarse_centroids: List[List[float]] = []
        self.codebooks: List[List[List[float]]] = []  # [sub_idx][centroid_idx] -> sub_vector
        self.inv_lists: Dict[int, List[Tuple[int, List[int], Dict[str, Any]]]] = {}
        self.is_trained = False

    def train(self, training_vectors: List[List[float]]):
        """Trains coarse Voronoi centroids and sub-space quantization codebooks."""
        # 1. Train coarse centroids using mini-batch k-means
        self.coarse_centroids = random.sample(training_vectors, min(self.nlist, len(training_vectors)))
        for c_id in range(len(self.coarse_centroids)):
            self.inv_lists[c_id] = []

        # 2. Train sub-vector codebooks
        self.codebooks = []
        for m in range(self.m_sub):
            start = m * self.sub_dim
            end = start + self.sub_dim
            sub_vecs = [v[start:end] for v in training_vectors]
            codebook = random.sample(sub_vecs, min(self.k_centroids, len(sub_vecs)))
            self.codebooks.append(codebook)

        self.is_trained = True

    def _quantize_vector(self, vec: List[float]) -> List[int]:
        codes = []
        for m in range(self.m_sub):
            start = m * self.sub_dim
            end = start + self.sub_dim
            sub = vec[start:end]
            best_idx = 0
            best_dist = float('inf')
            for c_idx, centroid in enumerate(self.codebooks[m]):
                d = euclidean_distance(sub, centroid)
                if d < best_dist:
                    best_dist = d
                    best_idx = c_idx
            codes.append(best_idx)
        return codes

    def add(self, vector_id: int, vector: List[float], payload: Dict[str, Any] = None):
        if not self.is_trained:
            raise RuntimeError("Index must be trained before inserting vectors")

        # Assign to nearest coarse Voronoi centroid
        best_cluster = 0
        best_dist = float('inf')
        for c_id, centroid in enumerate(self.coarse_centroids):
            d = euclidean_distance(vector, centroid)
            if d < best_dist:
                best_dist = d
                best_cluster = c_id

        codes = self._quantize_vector(vector)
        self.inv_lists[best_cluster].append((vector_id, codes, payload or {}))

    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        if not self.is_trained:
            return []

        # Step 1: Find nearest nprobe coarse Voronoi clusters
        cluster_dists = [(euclidean_distance(query, c), c_id) for c_id, c in enumerate(self.coarse_centroids)]
        cluster_dists.sort(key=lambda x: x[0])
        probe_clusters = [c_id for _, c_id in cluster_dists[:self.nprobe]]

        # Step 2: Compute Asymmetric Distance Lookup Table (ADC)
        # lut[m][centroid_idx] = distance from query sub-vector to codebook centroid
        lut = []
        for m in range(self.m_sub):
            start = m * self.sub_dim
            end = start + self.sub_dim
            q_sub = query[start:end]
            sub_table = [euclidean_distance(q_sub, c) ** 2 for c in self.codebooks[m]]
            lut.append(sub_table)

        # Step 3: Scan candidate entries in probed Voronoi cells using LUT lookup
        candidates = []
        for c_id in probe_clusters:
            for vid, codes, pl in self.inv_lists[c_id]:
                sq_dist = sum(lut[m][codes[m]] for m in range(self.m_sub))
                candidates.append((math.sqrt(sq_dist), vid, pl))

        candidates.sort(key=lambda x: x[0])
        return [(vid, d, pl) for d, vid, pl in candidates[:k]]
