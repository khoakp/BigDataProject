"""REC-01 Top-20 candidate, masking, and macro-metric contract."""
import math
import unittest

import torch

from src.evaluation.baselines import popularity_topk
from src.evaluation.rec01 import compare_recommenders
from src.evaluation.temporal import evaluate_topk, score_chunks


class Rec01EvaluatorTest(unittest.TestCase):
    def test_comparison_uses_one_warm_cohort_and_rejects_future_graph_edge(self):
        registry = [(0, "user", 7, 0), (1, "user", 99, 0)]
        registry += [(movie + 1, "movie", movie, movie - 1) for movie in range(1, 23)]
        train = [(i, i + 1, 99, i + 1, 5.) for i in range(22)]
        train += [(22, 23, 7, 1, 5.), (23, 24, 7, 2, 5.)]
        state = {"registry": registry,
                 "edges": [(1, movie + 1, movie - 1) for movie in range(1, 23)] +
                          [(0, 2, 22), (0, 3, 23)]}
        future = [(24, 25, 7, 3, 5.), (25, 26, 999, 4, 5.)]
        vectors = torch.ones((24, 1), dtype=torch.float32)
        result = compare_recommenders(vectors, state, train, future, chunk_size=4)
        self.assertEqual(result["cohort"]["warm_users"], 1)
        self.assertEqual(result["cohort"]["exclusions"]["cold_user"], 1)
        self.assertEqual(result["metrics"]["lightgcn"]["recall"], 1.)
        self.assertEqual(result["metrics"]["popularity"]["recall"], 1.)
        self.assertEqual(set(result["rankings"]["lightgcn"]), {7})
        self.assertEqual(set(result["rankings"]["popularity"]), {7})
        state["edges"].append((0, 4, 24))
        with self.assertRaisesRegex(ValueError, "Future edge"):
            compare_recommenders(vectors, state, train, future)

    def test_both_methods_use_same_twenty_unseen_train_movies(self):
        registry = [(0, "user", 7, 0), (1, "user", 99, 0)]
        registry += [(movie + 1, "movie", movie, movie) for movie in range(1, 23)]
        state = {"registry": registry,
                 "edges": [(1, movie + 1, movie) for movie in range(1, 23)] +
                          [(0, 2, 23), (0, 3, 24)]}
        users = {7: 0}
        movies = {movie: movie + 1 for movie in range(1, 23)}
        seen = {7: {1, 2}}
        vectors = torch.ones((24, 1), dtype=torch.float32)

        lightgcn = score_chunks(vectors, vectors, users, movies, seen, k=20, chunk_size=3)
        popularity = popularity_topk(state, users, movies, seen, k=20)
        expected = list(range(3, 23))
        self.assertEqual([movie for movie, _ in lightgcn[7]], expected)
        self.assertEqual([movie for movie, _ in popularity[7]], expected)

        truth = {7: {3, 22}}
        expected_ndcg = (1 + 1 / math.log2(21)) / (1 + 1 / math.log2(3))
        for ranking in (lightgcn, popularity):
            metrics = evaluate_topk(ranking, truth, 20)
            self.assertEqual(metrics["users"], 1)
            self.assertEqual(metrics["recall"], 1.0)
            self.assertAlmostEqual(metrics["ndcg"], expected_ndcg)


if __name__ == "__main__":
    unittest.main()
