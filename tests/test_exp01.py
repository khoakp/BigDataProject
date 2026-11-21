"""D11 toy contract: both measured methods export the same logical state."""
import tempfile
import unittest
import csv
from pathlib import Path

import duckdb

from src.benchmarking.exp01 import case_inputs, prepare_history, run_pilot, worker
from src.benchmarking.exp01_analysis import summarize
from src.graph.equivalence import compare_states, full_state


class Exp01Test(unittest.TestCase):
    def test_full_and_incremental_same_export_at_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "canonical.parquet"
            with duckdb.connect() as con:
                con.execute("CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
                con.executemany("INSERT INTO e VALUES (?,?,?,?,?,?)", [
                    (0, 1, 1, 10, 5., 0), (1, 2, 1, 11, 4., 1),
                    (2, 3, 2, 10, 2., 2), (3, 4, 2, 11, 5., 3)])
                con.execute(f"COPY e TO '{source.as_posix()}' (FORMAT PARQUET)")
            history = prepare_history(source, root / "history", 3)
            combined, delta = case_inputs(source, root / "case", 3, 1)
            full = worker("full", combined, delta, history["db_path"], root / "case/full")
            incremental = worker("incremental", combined, delta, history["db_path"], root / "case/incremental")
            self.assertEqual((full["events_processed"], incremental["events_processed"]), (4, 1))
            self.assertGreater(full["export_bytes_written"], 0)
            self.assertGreater(incremental["export_bytes_written"], 0)
            self.assertTrue(compare_states(full_state(root / "case/full"),
                                           full_state(root / "case/incremental"))["pass"])
            with duckdb.connect() as con:
                for table in ("registry", "edges", "degrees"):
                    schemas = [con.execute("DESCRIBE SELECT * FROM read_parquet(?)",
                                           [str(root / "case" / method / f"{table}.parquet")]).fetchall()
                               for method in ("full", "incremental")]
                    self.assertEqual(schemas[0], schemas[1])
            report = run_pilot(source, root / "pilot", root / "raw.csv", [(3, 1)], repetitions=2)
            self.assertEqual(report["status"], "PASS")
            with (root / "raw.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({int(r["repetition"]) for r in rows}, {0, 1, 2})
            self.assertTrue(all(r["correctness_pass"] == "True" for r in rows))
            _, summary = summarize(root / "raw.csv")
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["repetitions"], 2)
            self.assertGreater(summary[0]["paired_speedup_median"], 0)


if __name__ == "__main__":
    unittest.main()
