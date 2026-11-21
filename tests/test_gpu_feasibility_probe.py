"""Smoke the resource probe's actual model path on a tiny pre-cutoff graph."""
import tempfile
import unittest
from pathlib import Path

import duckdb

from scripts.gpu_feasibility_probe import run


class GpuFeasibilityProbeTest(unittest.TestCase):
    def test_cpu_probe_records_all_phases_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort = root / "cohort"
            cohort.mkdir()
            with duckdb.connect() as con:
                con.execute("CREATE TABLE nodes(node_id BIGINT,node_type VARCHAR,raw_id BIGINT)")
                con.executemany("INSERT INTO nodes VALUES (?,?,?)", [
                    (0, "user", 1), (1, "movie", 10), (2, "movie", 20),
                    (3, "user", 2), (4, "movie", 30)])
                con.execute("CREATE TABLE edges(user_node_id BIGINT,movie_node_id BIGINT,first_event_ts BIGINT)")
                con.executemany("INSERT INTO edges VALUES (?,?,?)", [
                    (0, 1, 1), (0, 2, 2), (3, 2, 3), (3, 4, 4)])
                con.execute(f"COPY nodes TO '{(cohort / 'nodes.parquet').as_posix()}' (FORMAT PARQUET)")
                con.execute(f"COPY edges TO '{(cohort / 'edges.parquet').as_posix()}' (FORMAT PARQUET)")
            output = root / "probe"
            result = run(cohort, output, device_name="cpu", dim=8, layers=1,
                         batch_size=4, steps=2, eval_users=2)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["graph"], {"nodes": 5, "positive_edges": 4,
                                               "sparse_entries": 8,
                                               "sparse_tensor_nnz": 8})
            self.assertEqual(result["ranking"]["users"], 2)
            self.assertEqual(result["ranking"]["candidate_movies"], 3)
            self.assertTrue((output / "probe_checkpoint.pt").is_file())
            for phase in ("load_python_graph", "normalize_python", "prepare_sampler",
                          "to_sparse_coo", "forward", "first_backward_adam_step",
                          "short_trajectory", "checkpoint", "evaluation_forward",
                          "rank_train_catalog"):
                self.assertIn(phase, result["phases"])
                self.assertGreaterEqual(result["phases"][phase]["wall_s"], 0)
            with self.assertRaises(FileExistsError):
                run(cohort, output, device_name="cpu")


if __name__ == "__main__":
    unittest.main()
