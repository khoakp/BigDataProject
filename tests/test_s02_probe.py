"""The S02 validation probe must not return the held-out test partition."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from src.evaluation import s02_probe


class S02ProbeTest(unittest.TestCase):
    def test_only_train_and_validation_rows_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "events.parquet"
            with duckdb.connect() as con:
                con.execute("CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating DOUBLE)")
                con.executemany("INSERT INTO e VALUES (?,?,?,?,?)", [
                    (0, 1, 1, 10, 5.), (1, 2, 1, 11, 5.),
                    (2, 3, 1, 12, 5.), (3, 4, 1, 13, 5.),
                    (4, 5, 1, 14, 5.),  # held-out test event
                ])
                con.execute(f"COPY e TO '{source.as_posix()}' (FORMAT PARQUET)")
            with (patch.object(s02_probe, "TRAIN_END_TS", 2),
                  patch.object(s02_probe, "VALID_END_TS", 4),
                  patch.object(s02_probe, "TRAIN_EVENTS", 2),
                  patch.object(s02_probe, "VALID_EVENTS", 2)):
                train, valid = s02_probe._approved_validation_events(source)
            self.assertEqual([row[0] for row in train], [0, 1])
            self.assertEqual([row[0] for row in valid], [2, 3])
            self.assertNotIn(4, [row[0] for row in train + valid])


if __name__ == "__main__":
    unittest.main()
