"""D10 must leave an honest failure manifest when GPU and fallback fail."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.d10_mvp import run


class D10Test(unittest.TestCase):
    def test_oom_then_failed_fallback_keeps_both_attempts(self):
        profile = {"source_sha256": "source", "prefixes": {"100000": {"sha256": "prefix"}}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            with patch("src.models.d10_mvp.canonicalize", return_value=profile), patch(
                "src.models.d10_mvp.pipeline", side_effect=[
                    torch.cuda.OutOfMemoryError("injected OOM"), RuntimeError("CPU failure")]):
                with self.assertRaisesRegex(RuntimeError, "CPU failure"):
                    run(Path(tmp) / "ratings.csv", Path(tmp) / "movies.csv", output,
                        device_name="cuda")
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ERROR")
            self.assertEqual([a["status"] for a in manifest["attempts"]], ["OOM", "ERROR"])
            self.assertEqual([a["device"] for a in manifest["attempts"]], ["cuda", "cpu"])
            self.assertEqual(manifest["config"]["prefix_events"], 100_000)

    def test_nonempty_output_is_not_a_clean_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            (output / "previous.txt").write_text("old", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Path(tmp) / "ratings.csv", Path(tmp) / "movies.csv", output)


if __name__ == "__main__":
    unittest.main()
