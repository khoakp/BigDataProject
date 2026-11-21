"""Create and verify a private, self-contained MovieGraph submission archive.

Selection is explicit: credentials, caches, raw MovieLens 32M and transient
graph branches are never swept into the archive by a broad recursive glob.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/s14_submission/MovieGraph-submission-S14.zip"
RUN = "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
BUNDLE = "data/processed/rec01_kaggle_bundle_s03"
FIXED = (
    "README.md", "AGENTS.md", "requirements-app.txt", "requirements-local-repro.txt",
    "data/processed/canonical_events/canonical_1000000.parquet",
    "data/processed/rec01_kaggle_runs/version_3/results.zip",
    "data/processed/s11_integration/rec01_1m_batch20_v1/report.json",
    "results/raw/exp01_full_vs_incremental.csv",
    "results/raw/exp01_scale2m_d12.csv",
    "results/raw/exp02_normalization.csv",
)
DOC_SUFFIXES = {".md", ".json", ".png", ".svg", ".docx", ".pdf", ".mmd"}
RUNME = """# MovieGraph submission package

The archive contains the source, 18-page IUH report (DOCX/PDF), defense Q&A,
REC-01 Kaggle bundle and sealed output, canonical 1M events, movies metadata,
benchmark CSV logs and S11 integration report. Raw MovieLens 32M is excluded.

Extract the archive into a private folder, open a PowerShell terminal in the
`BigDataProject` directory, and create a Python 3.10 virtual environment.
Install PyTorch for your CPU/CUDA platform, then install
`requirements-local-repro.txt`. See `docs/s12-reproducibility.md`.

```powershell
python -m scripts.rec01_kaggle_prep verify-output --bundle data/processed/rec01_kaggle_bundle_s03 --run data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M
$env:MOVIELENS_MOVIES_CSV = (Resolve-Path external/ml-32m/movies.csv).Path
python -m unittest discover -s tests -q
python -m streamlit run src/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

The app creates its own feedback graph branch under `data/processed/s10_feedback`.
The sealed REC-01 run remains unchanged. Open `docs/Bao-cao-MovieGraph-IUH.pdf`
and `docs/s13-defense-qa.md` for the report and defense notes.

The file `SUBMISSION-MANIFEST.json` lists SHA-256 and byte size for each member.
Run `python -m scripts.s14_package verify --archive <zip path>` to check the ZIP.

Before formal submission, the Project Owner must fill cover name/MSSV/class,
instructor, acknowledgments, signed declaration and actual group-work facts.
"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def files_to_include() -> dict[str, Path]:
    items: dict[str, Path] = {}

    def add(path: Path, archive_name: str | None = None) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        name = archive_name or path.relative_to(ROOT).as_posix()
        if name in items:
            raise ValueError(f"duplicate archive member: {name}")
        items[name] = path

    for name in FIXED:
        add(ROOT / name)
    for dirname in ("src", "scripts", "tests"):
        for path in sorted((ROOT / dirname).rglob("*.py")):
            add(path)
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        add(path)
    for path in sorted((ROOT / "docs").rglob("*")):
        if path.is_file() and path.suffix.lower() in DOC_SUFFIXES:
            add(path)
    for dirname in (RUN, BUNDLE):
        for path in sorted((ROOT / dirname).rglob("*")):
            if path.is_file():
                add(path)
    add(ROOT.parent / "ml-32m/movies.csv", "external/ml-32m/movies.csv")
    return items


def build(out: Path) -> dict:
    members = files_to_include()
    if len(members) < 100:
        raise ValueError("submission selection unexpectedly small")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                         cwd=ROOT, text=True).strip())
    manifest = {
        "schema": "s14-submission-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "git_worktree_dirty": dirty,
        "scope": "Private local package; contains derived MovieLens prefix/metadata, not raw 32M",
        "members": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            archive.writestr(name, data)
            manifest["members"][name] = {"bytes": len(data), "sha256": sha256_bytes(data)}
        runme = RUNME.encode("utf-8")
        archive.writestr("RUNME.md", runme)
        manifest["members"]["RUNME.md"] = {"bytes": len(runme), "sha256": sha256_bytes(runme)}
        archive.writestr("SUBMISSION-MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2,
                                    sort_keys=True).encode("utf-8") + b"\n")
    return verify(out)


def verify(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "SUBMISSION-MANIFEST.json" not in names:
            raise ValueError("duplicate member or missing manifest")
        manifest = json.loads(archive.read("SUBMISSION-MANIFEST.json"))
        expected = set(manifest["members"]) | {"SUBMISSION-MANIFEST.json"}
        if set(names) != expected:
            raise ValueError("archive file list differs from manifest")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"CRC failed: {bad}")
        for name, meta in manifest["members"].items():
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe member path: {name}")
            payload = archive.read(name)
            if len(payload) != meta["bytes"] or sha256_bytes(payload) != meta["sha256"]:
                raise ValueError(f"hash mismatch: {name}")
    return {"status": "PASS", "members": len(manifest["members"]),
            "archive_bytes": path.stat().st_size, "archive_sha256": sha256_bytes(path.read_bytes()),
            "git_worktree_dirty": manifest["git_worktree_dirty"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--out", type=Path, default=OUT)
    check = sub.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.out) if args.command == "build" else verify(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))
