"""Bundle validation and deterministic source archive for Kaggle REC-01."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import duckdb

from scripts.rec01_kaggle_prep import (inspect_source, seal_output, sha256,
                                       verify_output, write_source_zip)


class Rec01KagglePrepTest(unittest.TestCase):
    def test_sealed_output_detects_transfer_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, run = root / "bundle", root / "run"
            bundle.mkdir()
            run.mkdir()
            payload = run / "report.json"
            payload.write_text("test payload", encoding="utf-8")
            bundle_manifest = {"source_sha256": "source", "archive_sha256": "archive"}
            with patch("scripts.rec01_kaggle_prep.verify_input", return_value=bundle_manifest), patch(
                    "scripts.rec01_kaggle_prep._verify_run_core", return_value={"run_id": "toy"}):
                seal_output(bundle, run)
                self.assertEqual(verify_output(bundle, run)["run_id"], "toy")
                payload.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    verify_output(bundle, run)

    def test_source_protocol_check_on_toy_canonical_parquet(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "toy.parquet"
            with duckdb.connect() as con:
                con.execute("""CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,
                    movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)""")
                con.executemany("INSERT INTO e VALUES (?,?,?,?,?,?)",
                                [(i, i + 1, 1, i + 1, 5., i) for i in range(10)])
                con.execute("COPY e TO ? (FORMAT PARQUET)", [str(source)])
            expected = {"events": 10, "distinct_timestamps": 10, "train_end_ts": 8,
                        "valid_end_ts": 9, "train_events": 8, "valid_events": 1,
                        "test_events": 1}
            self.assertEqual(inspect_source(source, sha256(source), expected), expected)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                inspect_source(source, "incorrect", expected)

    def test_source_archive_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in ("src/models/rec01_train.py", "src/evaluation/rec01.py",
                             "src/evaluation/baselines.py", "src/evaluation/checkpoint_selection.py",
                             "scripts/rec01_kaggle_prep.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            first, second = root / "one.zip", root / "two.zip"
            hashes = write_source_zip(root, first)
            self.assertEqual(hashes, write_source_zip(root, second))
            self.assertEqual(sha256(first), sha256(second))
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(set(archive.namelist()), set(hashes))


if __name__ == "__main__":
    unittest.main()
