"""Train-only selection and post-cutoff coverage for GPU workload cohorts."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from src.evaluation.model_cohort import prepare_cohorts


CUTOFF = 1577836800


class ModelCohortTest(unittest.TestCase):
    def _source(self, path: Path, extra_future=()):
        rows = [
            (1, 10, 4, CUTOFF-5), (2, 10, 4, CUTOFF-4),
            (1, 20, 4, CUTOFF-3), (3, 20, 4, CUTOFF-2),
            (1, 30, 4, CUTOFF-1), (4, 30, 3, CUTOFF-1),
            (1, 20, 4, CUTOFF), (2, 20, 4, CUTOFF),
            (5, 10, 4, CUTOFF), (1, 10, 4, CUTOFF),
            (3, 20, 4, CUTOFF), (4, 30, 4, CUTOFF),
            (1, 30, 4, CUTOFF), (2, 40, 4, CUTOFF), *extra_future,
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("userId", "movieId", "rating", "timestamp"))
            writer.writerows(rows)

    def test_budget_ties_and_coverage_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ratings.csv"
            self._source(source)
            report = prepare_cohorts(source, root / "out", edge_budgets=(2, 4, 5),
                                     memory_limit="256MB", threads=1)
            a, b, c = report["cohorts"]
            self.assertEqual(report["base_positive_edges"], 5)
            self.assertEqual(report["future_positive_pairs"], 8)
            self.assertEqual((a["selected_rank_units"], a["positive_edges"], a["nodes"]), (1, 2, 3))
            self.assertEqual((b["selected_rank_units"], b["positive_edges"], b["evaluable_future_pairs"]),
                             (2, 4, 1))
            self.assertEqual((b["outside_model_user_pairs"],
                              b["outside_model_movie_pairs_after_user"],
                              b["train_seen_repeat_pairs"]), (2, 2, 3))
            self.assertEqual((a["cold_base_user_pairs"], a["omitted_base_user_pairs"]), (2, 1))
            self.assertEqual((b["cold_base_user_pairs"], b["omitted_base_user_pairs"]), (2, 0))
            self.assertEqual((b["cold_base_movie_pairs_after_user"],
                              b["omitted_base_movie_pairs_after_user"]), (1, 1))
            self.assertEqual((c["positive_edges"], c["movies"]), (5, 3))
            with duckdb.connect() as con:
                nodes = con.execute("SELECT node_id,node_type,raw_id FROM read_parquet(?) ORDER BY node_id",
                                    [str(root / "out/edges_2/nodes.parquet")]).fetchall()
            self.assertEqual(nodes, [(0, "user", 1), (1, "user", 2), (2, "movie", 10)])
            artifact = json.loads((root / "out/edges_2/manifest.json").read_text())
            self.assertEqual(artifact["source_sha256"], report["source_sha256"])
            self.assertEqual(artifact["cutoff_ts"], CUTOFF)
            self.assertEqual(artifact["graph_semantics"], "complete_induced_pre_cutoff_positive")
            self.assertFalse((root / "out/work.duckdb").exists())

    def test_future_changes_do_not_affect_selected_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a, source_b = root / "a.csv", root / "b.csv"
            self._source(source_a)
            self._source(source_b, extra_future=((9, 99, 5, CUTOFF+1),))
            for strategy in ("movie_degree", "user_degree", "user_recency",
                             "chronological_prefix", "hash_sample"):
                first = prepare_cohorts(source_a, root / (strategy + "_first"),
                                        edge_budgets=(4,), strategy=strategy,
                                        memory_limit="256MB", threads=1)
                second = prepare_cohorts(source_b, root / (strategy + "_second"),
                                         edge_budgets=(4,), strategy=strategy,
                                         memory_limit="256MB", threads=1)
                self.assertEqual(first["cohorts"][0]["nodes_sha256"],
                                 second["cohorts"][0]["nodes_sha256"])
                self.assertEqual(first["cohorts"][0]["edges_sha256"],
                                 second["cohorts"][0]["edges_sha256"])
                self.assertEqual(second["future_positive_pairs"], first["future_positive_pairs"]+1)


if __name__ == "__main__":
    unittest.main()
