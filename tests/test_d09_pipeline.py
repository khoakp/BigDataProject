import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.temporal import score_chunks
from src.metadata.movie_lookup import enrich_topk, load_movies


class D09Test(unittest.TestCase):
    def test_chunked_matches_independent_full_score_reference(self):
        users = torch.tensor([[1., 2.], [2., -1.]])
        movies = torch.tensor([[3., 1.], [0., 2.], [2., 2.], [-1., 4.]])
        user_ids = {40: 0, 41: 1}
        movie_ids = {9: 0, 10: 1, 11: 2, 12: 3}
        seen = {40: {9}, 41: {12}}
        actual = score_chunks(users, movies, user_ids, movie_ids, seen, k=2, chunk_size=1)
        for user, user_index in user_ids.items():
            reference = sorted(((float((users[user_index] * movies[index]).sum()), movie)
                                for movie, index in movie_ids.items() if movie not in seen[user]),
                               key=lambda pair: (-pair[0], pair[1]))[:2]
            self.assertEqual(actual[user], [(movie, score) for score, movie in reference])
        tied = score_chunks(torch.tensor([[1.]]), torch.tensor([[2.], [2.]]),
                            {1: 0}, {8: 0, 7: 1}, {}, k=2, chunk_size=1)
        self.assertEqual([movie for movie, _ in tied[1]], [7, 8])

    def test_left_lookup_preserves_rows_and_rejects_duplicate_keys(self):
        raw = [{"user_raw_id": 1, "movie_raw_id": 10, "rank": 1, "score": .5},
               {"user_raw_id": 1, "movie_raw_id": 99, "rank": 2, "score": .4}]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "movies.csv"
            source.write_text("movieId,title,genres\n10,Movie A,Drama\n11,,\n", encoding="utf-8")
            movies, profile = load_movies(source)
            enriched, coverage = enrich_topk(raw, movies)
            self.assertEqual(profile["missing_titles"], 1)
            self.assertEqual(coverage["missing_movie_ids"], 1)
            self.assertEqual(coverage["missing_titles"], 1)
            self.assertEqual(coverage["missing_genres"], 1)
            self.assertEqual(coverage["rows"], len(raw))
            self.assertEqual([tuple(row[key] for key in raw[0]) for row in enriched],
                             [tuple(row.values()) for row in raw])
            self.assertFalse(enriched[1]["metadata_found"])
            source.write_text("movieId,title,genres\n10,A,Drama\n10,B,Comedy\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_movies(source)

    def test_missing_metadata_fields_do_not_drop_or_reorder(self):
        raw = [{"user_raw_id": 2, "movie_raw_id": 11, "rank": 1, "score": 1.},
               {"user_raw_id": 2, "movie_raw_id": 12, "rank": 2, "score": .5}]
        enriched, coverage = enrich_topk(raw, {11: ("", None), 12: ("B", "Drama")})
        self.assertEqual(coverage, {"rows": 2, "metadata_found": 2,
                                    "missing_movie_ids": 0, "missing_titles": 1,
                                    "missing_genres": 1})
        self.assertEqual([(r["movie_raw_id"], r["rank"], r["score"]) for r in enriched],
                         [(11, 1, 1.), (12, 2, .5)])


if __name__ == "__main__":
    unittest.main()
