"""Offline checks for the private, one-device Kaggle probe bundle."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.kaggle_feasibility_runner import load_probe_plan, verify_mounted_files
from scripts.prepare_kaggle_feasibility import prepare, sha256


class KaggleFeasibilityBundleTest(unittest.TestCase):
    def test_optional_probe_plan_is_bounded_and_subset_only(self):
        available = ["edges_500000", "edges_2000000", "edges_5000000"]
        self.assertEqual(load_probe_plan(None, available), {
            "cohorts": available, "steps": 5, "source": "default"})
        self.assertEqual(load_probe_plan(None, available, {
            "cohorts": ["edges_5000000"], "steps": 500}), {
            "cohorts": ["edges_5000000"], "steps": 500, "source": "embedded"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe_plan.json"
            path.write_text(json.dumps({"cohorts": ["edges_5000000"], "steps": 500}),
                            encoding="utf-8")
            selected = load_probe_plan(path, available)
            self.assertEqual((selected["cohorts"], selected["steps"]),
                             (["edges_5000000"], 500))
            for invalid in ({"cohorts": ["edges_5000000"], "steps": 501},
                            {"cohorts": ["missing"], "steps": 5},
                            {"cohorts": ["edges_5000000", "edges_5000000"], "steps": 5},
                            {"cohorts": ["edges_5000000"], "steps": True}):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_probe_plan(path, available)

    def test_private_t4_bundle_has_verified_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort = root / "edges_4"
            cohort.mkdir()
            for name in ("nodes.parquet", "edges.parquet", "manifest.json"):
                (cohort / name).write_bytes(name.encode())
            output = root / "bundle"
            result = prepare([cohort], output, username="khoakp")
            dataset = Path(result["dataset"])
            kernel = Path(result["kernel"])
            metadata = json.loads((kernel / "kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["dataset_sources"], [result["dataset_id"]])
            self.assertTrue((kernel / "probe_runner.py").is_file())
            manifest = json.loads((dataset / "input_manifest.json").read_text())
            self.assertEqual(manifest["cohorts"], ["edges_4"])
            for name, expected in manifest["files_sha256"].items():
                self.assertEqual(sha256(dataset / name), expected)
            with zipfile.ZipFile(dataset / "project_source.zip") as archive:
                self.assertIn("scripts/gpu_feasibility_probe.py", archive.namelist())
            mounted = root / "kaggle_mount"
            mounted.mkdir()
            for archive_name in manifest["files_sha256"]:
                with zipfile.ZipFile(dataset / archive_name) as archive:
                    archive.extractall(mounted / archive_name.removesuffix(".zip"))
            verified, retained = verify_mounted_files(mounted, manifest)
            self.assertEqual(set(verified), set(manifest["unpacked_files_sha256"]))
            self.assertEqual(retained, [])
            unpacked_cohort = mounted / "edges_4/edges_4/nodes.parquet"
            unpacked_cohort.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Unpacked SHA-256 mismatch"):
                verify_mounted_files(mounted, manifest)
            with self.assertRaises(FileExistsError):
                prepare([cohort], output, username="khoakp")
            planned = prepare([cohort], root / "bundle_plan", username="khoakp",
                              probe_plan={"cohorts": ["edges_4"], "steps": 500})
            runner = (Path(planned["kernel"]) / "probe_runner.py").read_text(encoding="utf-8")
            self.assertIn('EMBEDDED_PROBE_PLAN = {"cohorts": ["edges_4"], "steps": 500}',
                          runner)


if __name__ == "__main__":
    unittest.main()
