"""End-to-end tests for portable REC-01 inference artifacts."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import duckdb
import torch

from src.graph.full_builder import file_sha256
from src.models.rec01_inference import load_rec01_inference
from src.models.rec01_train import Protocol, run


def _toy_source(path: Path) -> Protocol:
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


class Rec01InferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source = cls.root / "toy.parquet"
        protocol = _toy_source(source)
        run_dir = cls.root / "training_run"
        cls.training_report = run(source, run_dir, device_name="cpu", protocol=protocol)
        cls.portable = cls.root / "portable"
        cls.portable.mkdir()
        for name in ("report.json", "model_ids.json",
                     cls.training_report["selected_checkpoint_file"]):
            shutil.copy2(run_dir / name, cls.portable / name)
        shutil.copytree(run_dir / "model_graph", cls.portable / "model_graph")
        cls.movies_csv = cls.root / "movies.csv"
        with cls.movies_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("movieId", "title", "genres"))
            for movie_id in range(1, 22):
                writer.writerow((movie_id, f"Movie {movie_id}", "Drama"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _copied_bundle(self, target: Path) -> Path:
        shutil.copytree(self.portable, target)
        return target

    def test_portable_top20_masks_train_seen_then_enriches(self):
        self.assertFalse((self.portable / "graph_build").exists())
        service = load_rec01_inference(self.portable, self.movies_csv)
        result = service.recommend(7)
        self.assertEqual(result["method"], "lightgcn")
        self.assertEqual(result["checkpoint_step"],
                         self.training_report["selected_checkpoint"]["step"])
        self.assertEqual(result["model_graph_version"], 3)
        self.assertEqual(result["candidate_movies"], 22)
        self.assertEqual(result["train_seen_movies"], 2)
        self.assertEqual(len(result["items"]), 20)
        self.assertEqual([row["rank"] for row in result["items"]], list(range(1, 21)))
        self.assertEqual({row["movie_raw_id"] for row in result["items"]},
                         set(range(3, 23)))

        user_vector = service.embeddings[service.user_nodes[7]]
        exact = sorted(((movie, float(service.embeddings[node] @ user_vector))
                        for movie, node in service.movie_nodes.items() if movie not in (1, 2)),
                       key=lambda pair: (-pair[1], pair[0]))
        self.assertEqual([row["movie_raw_id"] for row in result["items"]],
                         [movie for movie, _ in exact])
        for row, (_, score) in zip(result["items"], exact):
            self.assertAlmostEqual(row["score"], score, places=6)
            if row["movie_raw_id"] == 22:
                self.assertIsNone(row["title"])
                self.assertFalse(row["metadata_found"])
            else:
                self.assertEqual(row["title"], f'Movie {row["movie_raw_id"]}')
                self.assertEqual(row["genres"], "Drama")
                self.assertTrue(row["metadata_found"])
        self.assertEqual(result["metadata_coverage"]["missing_movie_ids"], 1)
        self.assertEqual(result["metadata_coverage"]["rows"], 20)

    def test_cold_user_gets_train_only_popularity_and_exhausted_warm_user_is_empty(self):
        service = load_rec01_inference(self.portable, self.movies_csv)
        cold = service.recommend(123456)
        self.assertEqual(cold["method"], "popularity")
        self.assertIsNone(cold["checkpoint_step"])
        self.assertEqual(cold["train_seen_movies"], 0)
        self.assertEqual(cold["candidate_movies"], 22)
        self.assertEqual([row["movie_raw_id"] for row in cold["items"]],
                         list(range(1, 21)))
        self.assertEqual([row["score"] for row in cold["items"]],
                         [2.0, 2.0] + [1.0] * 18)
        self.assertEqual([row["rank"] for row in cold["items"]],
                         list(range(1, 21)))
        result = service.recommend(99)
        self.assertEqual(result["method"], "lightgcn")
        self.assertEqual(result["train_seen_movies"], 22)
        self.assertEqual(result["items"], [])

    def test_historical_source_hashes_are_provenance_not_runtime_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["code_sha256"]["src/evaluation/temporal.py"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            service = load_rec01_inference(bundle, self.movies_csv)
            self.assertEqual(service.recommend(7)["method"], "lightgcn")

    def test_checkpoint_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            checkpoint = bundle / self.training_report["selected_checkpoint_file"]
            with checkpoint.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
                load_rec01_inference(bundle, self.movies_csv)

    def test_checkpoint_shape_and_config_rejected_after_hash_update(self):
        for change, expected in (("shape", "size mismatch"),
                                 ("config", "embedding dimension")):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                bundle = self._copied_bundle(Path(tmp) / "bundle")
                report_path = bundle / "report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                step = report["selected_checkpoint"]["step"]
                checkpoint = bundle / report["selected_checkpoint_file"]
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if change == "shape":
                    saved["state_dict"]["embedding.weight"] = saved["state_dict"][
                        "embedding.weight"][:-1]
                else:
                    saved["config"]["dim"] += 1
                torch.save(saved, checkpoint)
                digest = file_sha256(checkpoint)
                report["selected_checkpoint"]["checkpoint_sha256"] = digest
                report["checkpoint_sha256"][str(step)] = digest
                for candidate in report["validation_candidates"]:
                    if candidate["step"] == step:
                        candidate["checkpoint_sha256"] = digest
                report_path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                    load_rec01_inference(bundle, self.movies_csv)

    def test_model_ids_must_match_graph_even_if_report_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            ids_path = bundle / "model_ids.json"
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            next(row for row in ids if row["node_type"] == "movie")["raw_id"] = 999
            ids_path.write_text(json.dumps(ids), encoding="utf-8")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["model_ids_sha256"] = file_sha256(ids_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model IDs and graph registry"):
                load_rec01_inference(bundle, self.movies_csv)

    def test_graph_version_must_match_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["graph"]["version"] += 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model graph version"):
                load_rec01_inference(bundle, self.movies_csv)

    def test_normalization_policy_rejected_after_outer_hash_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            manifest_path = bundle / "model_graph/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["normalization"] = "wrong_policy"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["graph"]["model_graph_manifest_sha256"] = file_sha256(manifest_path)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model graph policy mismatch"):
                load_rec01_inference(bundle, self.movies_csv)

    def test_unknown_artifact_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["artifact_schema"] = "future-format"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact schema"):
                load_rec01_inference(bundle, self.movies_csv)

    def test_protocol_config_hash_must_match_its_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["protocol"]["chunk_size"] += 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol config SHA-256"):
                load_rec01_inference(bundle, self.movies_csv)

    def test_selected_checkpoint_must_still_win_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._copied_bundle(Path(tmp) / "bundle")
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            selected_step = report["selected_checkpoint"]["step"]
            report["selected_checkpoint"]["validation_ndcg_20"] = 0.0
            report["selected_checkpoint"]["validation_recall_20"] = 0.0
            for candidate in report["validation_candidates"]:
                if candidate["step"] == selected_step:
                    candidate["validation_ndcg_20"] = 0.0
                    candidate["validation_recall_20"] = 0.0
                else:
                    candidate["validation_ndcg_20"] = 1.0
                    candidate["validation_recall_20"] = 1.0
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation-selected checkpoint"):
                load_rec01_inference(bundle, self.movies_csv)


if __name__ == "__main__":
    unittest.main()
