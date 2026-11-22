#!/usr/bin/env python3
"""
Benchmark Suite for Vector Similarity Search Engine (ANN)
Evaluates Recall@10 vs QPS (Queries Per Second) comparing HNSW, IVF-PQ, and Flat Brute-Force.
"""

import time
import random
import sys
import os

# Add root folder to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from ann_engine.flat_index import FlatIndex
from ann_engine.hnsw import HNSWIndex
from ann_engine.ivf_pq import IVFPQIndex

def generate_random_vectors(n: int, dim: int, seed: int = 42) -> List[List[float]]:
    random.seed(seed)
    return [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]

def compute_recall_at_k(approx_results, ground_truth_results, k=10) -> float:
    gt_ids = set(vid for vid, _, _ in ground_truth_results[:k])
    ann_ids = set(vid for vid, _, _ in approx_results[:k])
    overlap = len(gt_ids.intersection(ann_ids))
    return overlap / float(k)

def main():
    dim = 32
    num_docs = 2000
    num_queries = 100
    k = 10

    print("=" * 70)
    print("  HIGH-PERFORMANCE ANN VECTOR SIMILARITY ENGINE BENCHMARK")
    print(f"  Dataset: {num_docs} vectors | Dim: {dim} | Queries: {num_queries} | Top-K: {k}")
    print("=" * 70)

    print("[*] Generating high-dimensional vector dataset...")
    vectors = generate_random_vectors(num_docs, dim, seed=42)
    queries = generate_random_vectors(num_queries, dim, seed=999)

    # 1. Exact Flat Index (Ground Truth)
    print("\n[1/3] Building Flat Exact Index (Brute Force)...")
    flat = FlatIndex(metric="euclidean")
    for i, v in enumerate(vectors):
        flat.add(i, v, {"id": i, "category": f"cat_{i % 5}"})

    t0 = time.perf_counter()
    ground_truth = [flat.search(q, k=k) for q in queries]
    t_flat = (time.perf_counter() - t0) * 1000.0  # ms
    flat_qps = num_queries / (t_flat / 1000.0)

    # 2. HNSW Index
    print("[2/3] Building HNSW Graph Index (M=16, efConstruction=64)...")
    t0_build = time.perf_counter()
    hnsw = HNSWIndex(dim=dim, m=16, ef_construction=64, ef_search=32, metric="euclidean")
    for i, v in enumerate(vectors):
        hnsw.add(i, v, {"id": i, "category": f"cat_{i % 5}"})
    hnsw_build_time = (time.perf_counter() - t0_build) * 1000.0

    t0_query = time.perf_counter()
    hnsw_results = [hnsw.search(q, k=k) for q in queries]
    t_hnsw = (time.perf_counter() - t0_query) * 1000.0
    hnsw_qps = num_queries / (t_hnsw / 1000.0)
    hnsw_recall = sum(compute_recall_at_k(hnsw_results[i], ground_truth[i], k) for i in range(num_queries)) / num_queries

    # 3. IVF-PQ Index
    print("[3/3] Training and Indexing IVF-PQ (nlist=16, m_sub=4, nprobe=4)...")
    t0_build_pq = time.perf_counter()
    ivf_pq = IVFPQIndex(dim=dim, nlist=16, m_sub=4, nprobe=4)
    ivf_pq.train(vectors[:500])
    for i, v in enumerate(vectors):
        ivf_pq.add(i, v, {"id": i, "category": f"cat_{i % 5}"})
    ivf_pq_build_time = (time.perf_counter() - t0_build_pq) * 1000.0

    t0_query_pq = time.perf_counter()
    ivf_pq_results = [ivf_pq.search(q, k=k) for q in queries]
    t_ivf_pq = (time.perf_counter() - t0_query_pq) * 1000.0
    ivf_pq_qps = num_queries / (t_ivf_pq / 1000.0)
    ivf_pq_recall = sum(compute_recall_at_k(ivf_pq_results[i], ground_truth[i], k) for i in range(num_queries)) / num_queries

    # Output Benchmark Summary Table
    print("\n========================= BENCHMARK RESULTS =========================")
    print(f"{'Index Type':<16} | {'Build Time':<12} | {'Avg Latency':<12} | {'Throughput (QPS)':<18} | {'Recall@10':<10}")
    print("-" * 75)
    print(f"{'Flat (Exact)':<16} | {'0.0 ms':<12} | {t_flat/num_queries:.3f} ms/q   | {flat_qps:<18.1f} | {'100.0% (Oracle)':<10}")
    print(f"{'HNSW (Graph)':<16} | {hnsw_build_time:.1f} ms    | {t_hnsw/num_queries:.3f} ms/q   | {hnsw_qps:<18.1f} | {hnsw_recall * 100:.1f}%")
    print(f"{'IVF-PQ (Quant)':<16} | {ivf_pq_build_time:.1f} ms    | {t_ivf_pq/num_queries:.3f} ms/q   | {ivf_pq_qps:<18.1f} | {ivf_pq_recall * 100:.1f}%")
    print("=====================================================================\n")

if __name__ == "__main__":
    main()
