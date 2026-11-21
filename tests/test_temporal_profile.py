"""Boundary and graph-semantic checks for full-data temporal profiling."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.ingest.temporal_profile import profile, utc_second


class TemporalProfileTest(unittest.TestCase):
    def test_half_open_cutoff_new_entities_and_positive_edge_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ratings.csv"
            rows = [
                (1, 10, 3.0, utc_second("2018-01-01") - 1),
                (1, 10, 4.0, utc_second("2018-01-01")),
                (2, 20, 4.0, utc_second("2018-01-01")),
                (1, 10, 5.0, utc_second("2019-01-01")),
                (3, 20, 4.0, utc_second("2020-01-01")),
                (2, 30, 2.0, utc_second("2021-01-01")),
                (2, 30, 4.0, utc_second("2021-01-01")),
                (3, 20, 4.0, utc_second("2021-01-01")),
            ]
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("userId", "movieId", "rating", "timestamp"))
                writer.writerows(rows)
            output = root / "profile"
            result = profile(source, output, memory_limit="256MB", threads=1)
            self.assertEqual((result["rows"], result["positive_events"],
                              result["unique_positive_edges"]), (8, 6, 4))
            with (output / "cutoffs.csv").open(newline="", encoding="utf-8") as stream:
                cutoffs = {row["cutoff_utc"]: row for row in csv.DictReader(stream)}
            for day, base, replay, exact in (
                ("2018-01-01", 1, 7, 2),
                ("2019-01-01", 3, 5, 1),
                ("2020-01-01", 4, 4, 1),
                ("2021-01-01", 5, 3, 3),
            ):
                row = cutoffs[day]
                self.assertEqual((int(row["base_events"]), int(row["replay_events"]),
                                  int(row["events_exactly_at_cutoff"])),
                                 (base, replay, exact))
            c = cutoffs["2020-01-01"]
            self.assertEqual((int(c["base_graph_nodes"]), int(c["base_positive_edges"]),
                              int(c["replay_new_positive_edges"])), (4, 2, 2))
            self.assertEqual((int(c["replay_new_users"]), int(c["replay_new_movies"])), (1, 1))
            with (output / "monthly.csv").open(newline="", encoding="utf-8") as stream:
                months = list(csv.DictReader(stream))
            by_month = {row["month"]: row for row in months}
            self.assertEqual(int(by_month["2017-12"]["first_users"]), 1)
            self.assertEqual(int(by_month["2017-12"]["first_positive_graph_users"]), 0)
            self.assertEqual(int(by_month["2018-01"]["first_positive_graph_users"]), 2)
            self.assertEqual(int(months[-1]["cumulative_positive_edges"]), 4)
            self.assertEqual(int(months[-1]["cumulative_movies"]), 3)
            self.assertEqual(int(months[-1]["cumulative_positive_graph_movies"]), 3)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["cutoff_rule"],
                             "base < cutoff; replay >= cutoff")
            with (output / "yearly.csv").open(newline="", encoding="utf-8") as stream:
                years = {row["year"]: row for row in csv.DictReader(stream)}
            self.assertEqual(int(years["2018"]["distinct_users"]), 2)
            self.assertFalse((output / "profile.duckdb").exists())

    def test_refuses_existing_output_and_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ratings.csv"
            source.write_text("userId,movieId,rating,timestamp\n0,1,4,1514764800\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                profile(source, root / "invalid", memory_limit="256MB", threads=1)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                profile(source, existing, memory_limit="256MB", threads=1)


if __name__ == "__main__":
    unittest.main()
