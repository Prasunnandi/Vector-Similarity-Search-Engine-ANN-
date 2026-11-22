import unittest
from ann_engine.metrics import euclidean_distance, cosine_similarity, cosine_distance
from ann_engine.flat_index import FlatIndex
from ann_engine.hnsw import HNSWIndex
from ann_engine.ivf_pq import IVFPQIndex

class TestVectorEngine(unittest.TestCase):
    def test_metrics(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [1.0, 0.0, 0.0]

        self.assertAlmostEqual(euclidean_distance(v1, v3), 0.0)
        self.assertAlmostEqual(euclidean_distance(v1, v2), 1.41421356, places=5)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 1.0)
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)
        self.assertAlmostEqual(cosine_distance(v1, v3), 0.0)

    def test_flat_search(self):
        flat = FlatIndex(metric="euclidean")
        flat.add(1, [1.0, 2.0], {"title": "Doc 1"})
        flat.add(2, [10.0, 20.0], {"title": "Doc 2"})
        flat.add(3, [1.1, 2.1], {"title": "Doc 3"})

        results = flat.search([1.0, 2.0], k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][0], 3)

    def test_hnsw_recall(self):
        hnsw = HNSWIndex(dim=4, m=8, ef_construction=32, ef_search=16)
        vectors = [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.8, 0.7, 0.6],
            [0.11, 0.21, 0.31, 0.41],
            [5.0, 5.0, 5.0, 5.0]
        ]
        for i, vec in enumerate(vectors):
            hnsw.add(i, vec, {"label": f"vec_{i}"})

        # Query very close to vec 0 and vec 2
        res = hnsw.search([0.1, 0.2, 0.3, 0.4], k=2)
        top_ids = [vid for vid, _, _ in res]
        self.assertIn(0, top_ids)
        self.assertIn(2, top_ids)

    def test_ivf_pq_search(self):
        dim = 4
        ivf = IVFPQIndex(dim=dim, nlist=2, m_sub=2, nprobe=2)
        train_data = [
            [1.0, 1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1, 1.1],
            [9.0, 9.0, 9.0, 9.0],
            [9.1, 9.1, 9.1, 9.1]
        ]
        ivf.train(train_data)
        for i, v in enumerate(train_data):
            ivf.add(i, v, {"id": i})

        res = ivf.search([1.0, 1.0, 1.0, 1.0], k=2)
        self.assertEqual(len(res), 2)
        self.assertIn(res[0][0], [0, 1])

if __name__ == "__main__":
    unittest.main()
