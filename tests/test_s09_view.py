"""The S09 history view must never reveal validation/test interactions."""
import tempfile
import unittest
from pathlib import Path

import duckdb

from src.app.rec01_view import train_catalog, train_history, verify_history_source
from src.graph.full_builder import file_sha256


class S09ViewTest(unittest.TestCase):
    def test_train_history_cutoff_order_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "events.parquet"
            with duckdb.connect() as con:
                con.execute("CREATE TABLE events(event_id BIGINT,event_ts BIGINT,"
                            "user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,"
                            "source_row_id BIGINT)")
                con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", [
                    (0, 10, 5, 7, 4.5, 0), (1, 20, 5, 8, 3.0, 1),
                    (2, 20, 5, 9, 5.0, 2), (3, 21, 5, 10, 5.0, 3),
                ])
                con.execute(f"COPY events TO '{source.as_posix()}' (FORMAT PARQUET)")
            verify_history_source(source, file_sha256(source))
            with self.assertRaises(ValueError):
                verify_history_source(source, "0" * 64)
            result = train_history(source, 5, 20,
                                   {7: ("Seven", "Drama"), 9: ("Nine", "Comedy")},
                                   limit=2)
            self.assertEqual(result["total_train_ratings"], 3)
            self.assertEqual(result["positive_train_ratings"], 2)
            self.assertEqual([row["movie_raw_id"] for row in result["rows"]], [9, 8])
            self.assertEqual([row["positive_for_graph"] for row in result["rows"]],
                             [True, False])
            self.assertIsNone(result["rows"][1]["title"])
            self.assertNotIn(10, [row["movie_raw_id"] for row in result["rows"]])

    def test_catalog_is_train_movie_set_with_deterministic_order(self):
        rows = train_catalog({4: 2, 2: 0, 3: 1}, {2: 5, 3: 5, 4: 1},
                             {2: ("Two", "Drama"), 3: ("Three", "Comedy")})
        self.assertEqual([row["movie_raw_id"] for row in rows], [2, 3, 4])
        self.assertIsNone(rows[2]["title"])
        self.assertEqual(rows[0]["train_positive_users"], 5)


if __name__ == "__main__":
    unittest.main()
