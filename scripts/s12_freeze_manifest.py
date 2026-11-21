"""Capture source and evidence hashes for the S12 report freeze."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.graph.full_builder import file_sha256


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARTIFACTS = (
    "data/processed/canonical_events/canonical_1000000.parquet",
    "data/processed/rec01_kaggle_bundle_s03/input_manifest.json",
    "data/processed/rec01_kaggle_runs/version_3/results.zip",
    "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M/report.json",
    "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M/output_manifest.json",
    "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M/model_step_0300.pt",
    "results/raw/exp01_full_vs_incremental.csv",
    "results/raw/exp01_scale2m_d12.csv",
    "results/raw/exp02_normalization.csv",
    "data/processed/s11_integration/rec01_1m_batch20_v1/report.json",
)
PACKAGE_NAMES = ("duckdb", "torch", "numpy", "pandas", "pyarrow",
                 "streamlit", "psutil")


def _entry(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def build_manifest() -> dict:
    code = sorted((*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*.py"),
                   *ROOT.glob("tests/**/*.py")))
    report = ROOT / "docs/final-report.md"
    requirements = (ROOT / "requirements-app.txt",
                    ROOT / "requirements-local-repro.txt")
    figures = sorted((ROOT / "docs/report-assets").glob("*.png"))
    source_files = sorted(set((*code, report, *requirements, *figures)))
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                          cwd=ROOT, text=True).strip())
    return {
        "schema": "s12-freeze-v1", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head, "git_worktree_dirty": dirty,
        "environment": {"platform": platform.platform(),
                        "python": platform.python_version(), "packages": packages},
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): _entry(path)
                         for path in source_files},
        "evidence_artifacts": {name: _entry(ROOT / name) for name in REQUIRED_ARTIFACTS},
        "scope": "Hashes describe this local worktree and selected evidence; Git HEAD does not include uncommitted S01-S12 work",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "docs/s12-freeze-manifest.json")
    args = parser.parse_args()
    payload = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"schema": payload["schema"],
                      "source_files": len(payload["source_files"]),
                      "evidence_artifacts": len(payload["evidence_artifacts"]),
                      "git_worktree_dirty": payload["git_worktree_dirty"]}))
