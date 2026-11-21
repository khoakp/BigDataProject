"""Offline checks for the private, one-device Kaggle probe bundle."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.prepare_kaggle_feasibility import prepare, sha256


class KaggleFeasibilityBundleTest(unittest.TestCase):
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
            with self.assertRaises(FileExistsError):
                prepare([cohort], output, username="khoakp")


if __name__ == "__main__":
    unittest.main()
