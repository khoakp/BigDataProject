"""Independent toy expectations for the D03 full rebuild reference."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE = Path(__file__).resolve().parents[1] / "src" / "graph" / "full_builder.py"
spec = importlib.util.spec_from_file_location("full_builder", MODULE)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FullBuilderTest(unittest.TestCase):
    def test_positive_dedup_registry_and_degrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            events = base / "events.parquet"
            db = duckdb.connect()
            db.execute("CREATE TABLE t(event_id BIGINT, event_ts BIGINT, user_raw_id BIGINT, movie_raw_id BIGINT, rating FLOAT, source_row_id BIGINT)")
            db.executemany("INSERT INTO t VALUES (?,?,?,?,?,?)", [
                (0, 10, 10, 50, 3.5, 0),  # filtered, no node allocation
                (1, 11, 10, 50, 4.0, 1),
                (2, 12, 20, 50, 5.0, 2),
                (3, 13, 10, 50, 4.5, 3),  # duplicate positive edge
                (4, 14, 20, 60, 4.0, 4),
            ])
            db.execute(f"COPY t TO '{events.as_posix()}' (FORMAT PARQUET)")
            result = builder.rebuild(events, base / "state")
            self.assertEqual((result["positive_events"], result["edges"], result["nodes"]), (4, 3, 4))
            self.assertEqual((result["sum_user_degree"], result["sum_movie_degree"]), (3, 3))
            registry = db.execute(f"SELECT node_id,node_type,raw_id,first_event_id FROM read_parquet('{(base / 'state' / 'registry.parquet').as_posix()}') ORDER BY node_id").fetchall()
            self.assertEqual(registry, [(0, 'user', 10, 1), (1, 'movie', 50, 1), (2, 'user', 20, 2), (3, 'movie', 60, 4)])
            degrees = db.execute(f"SELECT node_id,degree FROM read_parquet('{(base / 'state' / 'degrees.parquet').as_posix()}') ORDER BY node_id").fetchall()
            self.assertEqual(degrees, [(0, 1), (1, 2), (2, 2), (3, 1)])


if __name__ == "__main__":
    unittest.main()
