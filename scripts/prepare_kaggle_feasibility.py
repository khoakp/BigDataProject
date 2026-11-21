"""Package fixed pre-cutoff cohorts and probe source for a private Kaggle run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "moviegraph-v2-feasibility-input"
KERNEL_SLUG = "moviegraph-v2-gpu-feasibility"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add(archive: zipfile.ZipFile, source: Path, member: str) -> None:
    entry = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(entry, source.read_bytes())


def prepare(cohorts: list[Path], output: Path, *, username: str) -> dict:
    if not cohorts or not username or "/" in username:
        raise ValueError("Expected cohorts and Kaggle username")
    cohorts = [Path(path).resolve() for path in cohorts]
    for cohort in cohorts:
        if not all((cohort / name).is_file() for name in
                   ("nodes.parquet", "edges.parquet", "manifest.json")):
            raise FileNotFoundError(f"Incomplete cohort: {cohort}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "dataset"
    kernel = output / "kernel"
    dataset.mkdir()
    kernel.mkdir()

    source_zip = dataset / "project_source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for path in sorted((ROOT / "src").rglob("*.py")):
            _add(archive, path, path.relative_to(ROOT).as_posix())
        _add(archive, ROOT / "scripts/gpu_feasibility_probe.py",
             "scripts/gpu_feasibility_probe.py")

    names = []
    for cohort in cohorts:
        name = cohort.name
        if name in names:
            raise ValueError("Cohort directory names must be unique")
        names.append(name)
        target = dataset / f"{name}.zip"
        with zipfile.ZipFile(target, "w") as archive:
            for filename in ("nodes.parquet", "edges.parquet", "manifest.json"):
                _add(archive, cohort / filename, f"{name}/{filename}")

    input_files = (source_zip.name,) + tuple(f"{name}.zip" for name in names)
    input_manifest = {"dataset_id": f"{username}/{DATASET_SLUG}",
                      "cohorts": names,
                      "files_sha256": {name: sha256(dataset / name) for name in input_files}}
    (dataset / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2),
                                                 encoding="utf-8")
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "MovieGraph v2 GPU Feasibility Input",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
        "description": "Private derived MovieLens 32M pre-2020 positive graph cohorts for GPU feasibility. Original MovieLens terms apply; not a public training release.",
    }, indent=2), encoding="utf-8")

    shutil.copy2(ROOT / "scripts/kaggle_feasibility_runner.py", kernel / "probe_runner.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "MovieGraph v2 GPU Feasibility",
        "code_file": "probe_runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{username}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2), encoding="utf-8")
    return {"dataset": str(dataset), "kernel": str(kernel), **input_manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cohort, args.out, username=args.username), indent=2))


if __name__ == "__main__":
    main()
