# Vector-Similarity-Search-Engine-ANN-

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Recall@10](https://img.shields.io/badge/Recall%4010-96.1%25-brightgreen.svg)]()
[![Throughput](https://img.shields.io/badge/Throughput-1%2C334%20QPS-purple.svg)]()
[![Algorithms](https://img.shields.io/badge/Algorithms-HNSW%20%7C%20IVF--PQ%20%7C%20Flat-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

A high-performance Approximate Nearest Neighbors (ANN) vector similarity search engine implemented from scratch. Supports **Hierarchical Navigable Small World (HNSW)** multi-layer graphs, **Inverted File with Product Quantization (IVF-PQ)** for 16x vector memory compression, and brute-force Exact Flat indexing as an empirical ground-truth baseline.

---

## 🏛️ Architecture & Index Comparison

```
                         HIGH-DIMENSIONAL QUERY VECTOR (D=128)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │         HNSW GRAPH          │               │      IVF-PQ QUANTIZATION    │
     │  Multi-Layer Skip-Graph     │               │  Voronoi Coarse Centroids   │
     │  $O(\log N)$ Greedy Search  │               │  16x Memory Sub-Space Codes │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
                    │ Beam Search (efSearch=32)                   │ Asymmetric Distance (ADC)
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │   Recall@10: 96.1%          │               │   Recall@10: ~92% (High Ctr)│
     │   Latency: 3.0 ms / query   │               │   Latency: 0.75 ms / query  │
     │   High Recall Workloads     │               │   Ultra-Scale Billion Vector│
     └─────────────────────────────┘               └─────────────────────────────┘
```

---

## ⚡ Algorithms Implemented

### 1. Hierarchical Navigable Small World (HNSW)
- **Logarithmic Layer Scaling:** Elements assigned to layer $l$ with probability $P(l) = e^{-l \cdot m_L}$ where $m_L = 1 / \ln(M)$.
- **Greedy Multi-Layer Beam Search:** Fast top-down entry-point descent across upper sparse layers, transitioning to fine-grained local neighbor expansion at layer 0.
- **Bi-Directional Heuristic Edge Truning:** Enforces bounded degree $M$ and $M_0 = 2M$ to avoid graph clutter.

### 2. Inverted File with Product Quantization (IVF-PQ)
- **Voronoi Coarse Clustering:** $k$-means partitioning into `nlist` cells to restrict search to candidate `nprobe` cells.
- **Sub-Vector Quantization:** Breaks $D$-dimensional vector into $M$ sub-vectors of size $D/M$, quantizing each to a 1-byte code index.
- **Asymmetric Distance Computation (ADC):** Pre-computes query-to-centroid distance lookup tables (LUT) for zero inner loop floating-point arithmetic.

---

## 📊 Benchmark Results

Evaluated across 2,000 vectors with 32 dimensions on 100 queries:

| Index Type | Build Time | Query Latency | Throughput (QPS) | Recall@10 | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flat (Exact)** | **0.0 ms** | 11.359 ms/q | 88.0 QPS | **100.0% (Oracle)** | Baseline (1.0x) |
| **HNSW (Graph)** | **16.6 s** | 3.008 ms/q | 332.4 QPS | **96.1%** | 1.8x (Graph Edges) |
| **IVF-PQ (Quant)** | **352.0 ms**| **0.750 ms/q** | **1,334.1 QPS** | **~90.0%** | **16x Compressed** |

---

## 🚀 Quickstart

### 1. Run Unit Tests
```bash
python -m unittest tests/test_vector_engine.py
```

### 2. Run Comprehensive Performance Benchmark
```bash
python benchmarks/run_benchmark.py
```

### 3. Open Interactive 2D Vector Space Visualizer
Open `web/index.html` in your browser to interactively click and explore Euclidean / Cosine nearest neighbor radii in real time.

---

## 📁 Repository Directory Structure

```
├── ann_engine/
│   ├── flat_index.py            # Exact k-NN brute-force ground truth oracle
│   ├── hnsw.py                  # Hierarchical Navigable Small World graph index
│   ├── ivf_pq.py                # Inverted File Product Quantization index
│   └── metrics.py               # Euclidean, Cosine Similarity, and Dot Product
├── benchmarks/
│   └── run_benchmark.py         # Recall@K vs QPS benchmark runner
├── tests/
│   └── test_vector_engine.py    # Unit tests
├── web/
│   └── index.html               # Interactive vector projection UI
└── README.md                    # Technical documentation
```

---

## 📜 License
MIT License. Created by [Prasun Nandi](https://github.com/Prasunnandi).
