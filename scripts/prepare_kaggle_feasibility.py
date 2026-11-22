"""Package fixed pre-cutoff cohorts and probe source for a private Kaggle run."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.kaggle_feasibility_runner import load_probe_plan


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


def prepare(cohorts: list[Path], output: Path, *, username: str,
            probe_plan: dict | None = None) -> dict:
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

    unpacked_files_sha256 = {}
    source_zip = dataset / "project_source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for path in sorted((ROOT / "src").rglob("*.py")):
            member = path.relative_to(ROOT).as_posix()
            _add(archive, path, member)
            unpacked_files_sha256[f"project_source/{member}"] = sha256(path)
        probe_source = ROOT / "scripts/gpu_feasibility_probe.py"
        _add(archive, probe_source, "scripts/gpu_feasibility_probe.py")
        unpacked_files_sha256["project_source/scripts/gpu_feasibility_probe.py"] = sha256(probe_source)

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
                unpacked_files_sha256[f"{name}/{name}/{filename}"] = sha256(cohort / filename)

    input_files = (source_zip.name,) + tuple(f"{name}.zip" for name in names)
    input_manifest = {"dataset_id": f"{username}/{DATASET_SLUG}",
                      "cohorts": names,
                      "files_sha256": {name: sha256(dataset / name) for name in input_files},
                      "unpacked_files_sha256": unpacked_files_sha256}
    (dataset / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2),
                                                 encoding="utf-8")
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "MovieGraph v2 GPU Feasibility Input",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
        "description": "Private derived MovieLens 32M pre-2020 positive graph cohorts for GPU feasibility. Original MovieLens terms apply; not a public training release.",
    }, indent=2), encoding="utf-8")

    runner_source = (ROOT / "scripts/kaggle_feasibility_runner.py").read_text(encoding="utf-8")
    if probe_plan is not None:
        validated = load_probe_plan(None, names, probe_plan)
        marker = "EMBEDDED_PROBE_PLAN = None"
        if runner_source.count(marker) != 1:
            raise ValueError("Expected exactly one embedded probe plan marker")
        inline_plan = json.dumps({"cohorts": validated["cohorts"],
                                  "steps": validated["steps"]})
        runner_source = runner_source.replace(
            marker, f"EMBEDDED_PROBE_PLAN = {inline_plan}")
    (kernel / "probe_runner.py").write_bytes(runner_source.encode("utf-8"))
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
    parser.add_argument("--plan", type=Path,
                        help="Optional bounded {cohorts,steps} plan embedded in Kaggle's code_file")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else None
    print(json.dumps(prepare(args.cohort, args.out, username=args.username,
                             probe_plan=plan), indent=2))


if __name__ == "__main__":
    main()
