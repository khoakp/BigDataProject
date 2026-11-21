"""Popularity uses only train positives and the shared warm-candidate contract."""
import unittest

from src.evaluation.baselines import popularity_topk
from src.evaluation.temporal import evaluate_topk


class PopularityBaselineTest(unittest.TestCase):
    def setUp(self):
        # Node order deliberately differs from raw movie ID order.
        self.state = {
            "registry": [(0, "user", 100, 0), (1, "movie", 30, 0),
                         (2, "movie", 10, 1), (3, "user", 200, 2),
                         (4, "movie", 20, 3), (5, "user", 300, 4)],
            "edges": [(0, 1, 0), (0, 2, 1), (3, 2, 2),
                      (3, 4, 3), (5, 2, 4), (5, 4, 5)],
        }
        self.users = {100: 0, 200: 3}
        self.movies = {30: 1, 10: 2, 20: 4}

    def test_train_counts_mask_and_shared_metric_contract(self):
        # Train counts are 10 -> 3, 20 -> 2, 30 -> 1.
        ranked = popularity_topk(self.state, self.users, self.movies,
                                 {100: {10, 30}, 200: {10, 20}}, k=20)
        self.assertEqual(ranked, {100: [(20, 2.0)],
                                  200: [(30, 1.0)]})
        self.assertEqual(evaluate_topk(ranked, {100: {20}, 200: {30}}, 20)["recall"], 1.0)

    def test_ties_use_raw_movie_id_and_duplicate_edge_does_not_inflate(self):
        self.state["edges"].append((3, 1, 6))  # 30 and 20 now have two distinct users.
        self.state["edges"].append((3, 1, 7))  # Duplicate pair contributes no extra user.
        ranked = popularity_topk(self.state, {200: 3}, self.movies, {}, k=3)
        self.assertEqual(ranked[200], [(10, 3.0), (20, 2.0), (30, 2.0)])

    def test_rejects_future_catalog_or_invalid_k(self):
        with self.assertRaisesRegex(ValueError, "train graph registry"):
            popularity_topk(self.state, self.users, {**self.movies, 99: 99}, {}, k=20)
        with self.assertRaisesRegex(ValueError, "complete train graph catalog"):
            popularity_topk(self.state, self.users, {10: 2, 20: 4}, {}, k=20)
        with self.assertRaisesRegex(ValueError, "k must be positive"):
            popularity_topk(self.state, self.users, self.movies, {}, k=0)


if __name__ == "__main__":
    unittest.main()
