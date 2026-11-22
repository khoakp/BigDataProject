"""Toy ordering and deterministic replay check for D02."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import duckdb


MODULE = Path(__file__).resolve().parents[1] / "src" / "ingest" / "d02_profile.py"
spec = importlib.util.spec_from_file_location("d02_profile", MODULE)
d02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d02)


class D02Test(unittest.TestCase):
    def test_rejects_nonpositive_ids_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "ratings.csv"
            source.write_text("userId,movieId,rating,timestamp\n0,10,4,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source validation"):
                d02.run(source, base / "invalid", max_prefix=1)
            source.write_text("userId,movieId,rating,timestamp\n1,10,4,1\n", encoding="utf-8")
            d02.run(source, base / "valid", max_prefix=1)
            before = (base / "valid/canonical_1.parquet").read_bytes()
            source.write_text("userId,movieId,rating,timestamp\n2,10,4,1\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                d02.run(source, base / "valid", max_prefix=1)
            self.assertEqual((base / "valid/canonical_1.parquet").read_bytes(), before)
    def test_ties_and_identical_rows_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "ratings.csv"
            # Physical source rows: timestamps tie, and the last two rows are identical.
            source.write_text("userId,movieId,rating,timestamp\n2,9,4.0,20\n1,8,3.0,10\n1,8,3.0,10\n2,9,5.0,10\n", encoding="utf-8")
            a = d02.run(source, base / "a", max_prefix=4)
            b = d02.run(source, base / "b", max_prefix=4)
            self.assertEqual(a["rows"], 4)
            self.assertEqual(a["duplicate_pair_rows"], 2)
            self.assertEqual(a["timestamp_tie_rows"], 2)
            self.assertEqual(a["prefixes"]["4"]["sha256"], b["prefixes"]["4"]["sha256"])
            rows = duckdb.sql(f"SELECT event_id, source_row_id FROM read_parquet('{(base / 'a' / 'canonical_4.parquet').as_posix()}') ORDER BY event_id").fetchall()
            self.assertEqual(rows, [(0, 1), (1, 2), (2, 3), (3, 0)])


if __name__ == "__main__":
    unittest.main()
