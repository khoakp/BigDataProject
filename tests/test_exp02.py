"""D13 cadence and common-boundary normalization on a tiny canonical replay."""
import tempfile
import unittest
from pathlib import Path

import duckdb

from src.benchmarking.exp02 import compare_common, prepare_batches, worker


class Exp02Test(unittest.TestCase):
    def test_eager_deferred_freshness_and_common_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "canonical.parquet"
            with duckdb.connect() as con:
                con.execute("CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
                con.executemany("INSERT INTO e VALUES (?,?,?,?,?,?)", [
                    (i, i+1, i//5+1, i%5+1, 5., i) for i in range(10)])
                con.execute(f"COPY e TO '{source.as_posix()}' (FORMAT PARQUET)")
            batches = prepare_batches(source, root / "batches", 2, 10)
            provenance = {"dataset_id": "toy", "canonical_input_sha256": "toy"}
            eager_dir, deferred_dir = root / "r1", root / "r5"
            eager = worker(batches, eager_dir, 1, provenance, "toy", "measured", 1)
            deferred = worker(batches, deferred_dir, 5, provenance, "toy", "measured", 1)
            self.assertEqual([r["staleness_batches"] for r in eager], [0]*5)
            self.assertEqual([r["staleness_batches"] for r in deferred], [1, 2, 3, 4, 0])
            self.assertEqual([r["staleness_events"] for r in deferred], [2, 4, 6, 8, 0])
            self.assertEqual(deferred[-1]["snapshot_count"], 1)
            self.assertTrue(compare_common(eager_dir, deferred_dir, 5)["pass"])


if __name__ == "__main__":
    unittest.main()
