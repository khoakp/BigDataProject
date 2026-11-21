import unittest
import tempfile
from pathlib import Path

import duckdb
import torch

from src.evaluation.temporal import cutoff_cohort, evaluate_topk, score_chunks, split_events
from src.models.d08_baseline import choose_cutoffs, train_only_graph, sample_batches


class TemporalTest(unittest.TestCase):
    def test_small_timestamp_cutoffs_stay_distinct(self):
        events = [(i, i, 1, i, 5.) for i in range(3)]
        self.assertEqual(choose_cutoffs(events), (0, 1))

    def test_cutoff_and_cold_exclusions(self):
        events = [(0, 1, 1, 10, 5.), (1, 2, 1, 11, 4.),
                  (2, 3, 1, 12, 5.), (3, 3, 2, 10, 5.),
                  (4, 4, 1, 11, 5.), (5, 5, 1, 10, 4.)]
        train, valid, test = split_events(events, 2, 4)
        self.assertEqual([e[0] for e in train], [0, 1])
        registry = [(0, 'user', 1, 0), (1, 'movie', 10, 0), (2, 'movie', 11, 1)]
        truth, seen, cohort = cutoff_cohort(train, valid, registry)
        self.assertEqual(truth, {})
        self.assertEqual(cohort['exclusions'], {'cold_movie': 1, 'cold_user': 1, 'train_seen_repeat': 1, 'no_candidates_users': 0})
        self.assertEqual(seen[1], {10, 11})
        self.assertEqual([e[0] for e in test], [5])

    def test_chunked_ranking_and_metrics(self):
        vectors = torch.tensor([[1., 0.], [0., 1.], [2., 0.], [3., 0.]])
        ranked = score_chunks(vectors, vectors, {1: 0}, {10: 1, 11: 2, 12: 3}, {1: {10}}, 2, 1)
        self.assertEqual([movie for movie, _ in ranked[1]], [12, 11])
        self.assertEqual(ranked, score_chunks(vectors, vectors, {1: 0},
                                              {10: 1, 11: 2, 12: 3}, {1: {10}}, 2, 3))
        self.assertEqual(evaluate_topk(ranked, {1: {11, 12}}, 2), {'users': 1, 'recall': 1., 'ndcg': 1.})

    def test_future_events_never_enter_graph_or_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'events.parquet'
            with duckdb.connect() as con:
                con.execute("CREATE TABLE events(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
                con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", [
                    (0, 1, 1, 10, 5., 0), (1, 2, 1, 11, 5., 1),
                    (2, 2, 2, 10, 5., 2), (3, 3, 2, 12, 5., 3),
                    (4, 4, 1, 12, 5., 4)])
                con.execute(f"COPY events TO '{source.as_posix()}' (FORMAT PARQUET)")
            summary, state, adjacency = train_only_graph(source, Path(tmp), 3)
            self.assertEqual(summary['edges'], 3)
            self.assertEqual({r[2] for r in state['registry'] if r[1] == 'movie'}, {10, 11})
            self.assertEqual(adjacency.shape, (4, 4))
            train_movie_nodes = {r[0] for r in state['registry'] if r[1] == 'movie'}
            for batch in sample_batches(state, 16, 2, 7):
                self.assertTrue(all(negative in train_movie_nodes for _, _, negative in batch))


if __name__ == '__main__':
    unittest.main()
