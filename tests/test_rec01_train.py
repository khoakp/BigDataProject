"""Small REC-01 training run validates split, selection, reload, and artifacts."""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import duckdb

from src.evaluation.checkpoint_selection import select_checkpoint
from src.evaluation.rec01 import compare_recommenders
from src.graph.full_builder import file_sha256
from src.models.d08_baseline import sample_batches
from src.models.rec01_train import Protocol, _load_partition, _sample_batches, run


def toy_source(path: Path) -> Protocol:
    # First 24 events are train. User 7 sees items 1/2; user 99 introduces 1..22.
    rows = [(i, i + 1, 99, i + 1, 5., i) for i in range(22)]
    rows += [(22, 23, 7, 1, 5., 22), (23, 24, 7, 2, 5., 23)]
    rows += [(i, i + 1, 7, i - 21, 5., i) for i in range(24, 30)]
    with duckdb.connect() as con:
        con.execute("CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,"
                    "movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
        con.executemany("INSERT INTO e VALUES (?,?,?,?,?,?)", rows)
        con.execute(f"COPY e TO '{path.as_posix()}' (FORMAT PARQUET)")
    return Protocol(source_sha256=file_sha256(path), events=30, distinct_timestamps=30,
                    train_end_ts=24, valid_end_ts=27,
                    train_events=24, valid_events=3, test_events=3,
                    checkpoint_steps=(1, 2), batch_size=4, dim=4, layers=1,
                    graph_batch_events=10, chunk_size=4)


class Rec01TrainTest(unittest.TestCase):
    def test_cached_sampler_preserves_d08_sequence(self):
        state = {"registry": [(0, "user", 1, 0), (1, "movie", 10, 0),
                              (2, "movie", 11, 0), (3, "movie", 12, 0),
                              (4, "user", 2, 0)],
                 "edges": [(0, 1, 0), (0, 2, 1), (4, 3, 2)]}
        self.assertEqual(list(_sample_batches(state, 8, 5, 7)),
                         list(sample_batches(state, 8, 5, 7)))

    def test_toy_run_selects_before_test_and_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "toy.parquet"
            protocol = toy_source(source)
            selected = False
            test_calls = 0
            test_loads = 0

            def selection_spy(records):
                nonlocal selected
                self.assertTrue(all("test" not in str(record).lower() for record in records))
                selected = True
                return select_checkpoint(records)

            def comparison_spy(embeddings, state, train, future, **kwargs):
                nonlocal test_calls
                if min(event[0] for event in future) >= 27:
                    self.assertTrue(selected, "Test evaluated before checkpoint selection")
                    test_calls += 1
                return compare_recommenders(embeddings, state, train, future, **kwargs)

            def partition_spy(path, valid_end_ts, *, test):
                nonlocal test_loads
                if test:
                    self.assertTrue(selected, "Test labels loaded before checkpoint selection")
                    test_loads += 1
                return _load_partition(path, valid_end_ts, test=test)

            with patch("src.models.rec01_train.select_checkpoint", side_effect=selection_spy), \
                 patch("src.models.rec01_train.compare_recommenders", side_effect=comparison_spy), \
                 patch("src.models.rec01_train._load_partition", side_effect=partition_spy):
                report = run(source, root / "run", device_name="cpu", protocol=protocol)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(test_calls, 1)
            self.assertEqual(test_loads, 1)
            self.assertEqual(len(report["validation_candidates"]), 2)
            self.assertIn(report["selected_checkpoint"]["step"], (1, 2))
            self.assertTrue(report["selected_checkpoint_reloaded"])
            self.assertEqual(report["graph"]["version"], 3)
            self.assertEqual(report["metrics"]["test"]["lightgcn"]["users"], 1)
            self.assertEqual(report["metrics"]["test"]["popularity"]["users"], 1)
            for step in (1, 2):
                self.assertTrue((root / "run" / f"model_step_{step:04d}.pt").exists())
            self.assertTrue((root / "run" / "model_ids.json").exists())
            self.assertTrue((root / "run" / "model_graph" / "manifest.json").exists())
            self.assertTrue((root / "run" / "rankings.json").exists())
            with (root / "run" / "rec01_quality.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)  # 2 val checkpoints + val Popularity + test pair
            self.assertEqual(sum(row["split"] == "test" for row in rows), 2)
            self.assertTrue(all(row["status"] == "PASS" for row in rows))
            saved_report = json.loads((root / "run" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_report["status"], "PASS")

    def test_wrong_source_hash_fails_before_graph_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "toy.parquet"
            protocol = replace(toy_source(source), source_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run(source, root / "run", device_name="cpu", protocol=protocol)
            report = json.loads((root / "run" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ERROR")
            self.assertFalse((root / "run" / "graph_build").exists())


if __name__ == "__main__":
    unittest.main()
