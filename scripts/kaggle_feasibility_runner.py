"""Kaggle T4 script: verify inputs, then probe one GPU (cuda:0) by cohort.

The Kaggle metadata attaches a private dataset made by prepare_kaggle_feasibility.
Each cohort runs in its own Python process so CPU/GPU peaks are independent.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_verified(archive_path: Path, target: Path, expected: set[str] | None = None) -> None:
    """Extract only regular files within target after validating member names."""
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = {member.filename for member in members}
        if expected is not None and names != expected:
            raise ValueError(f"Unexpected archive members in {archive_path.name}")
        for member in members:
            name = member.filename
            parts = Path(name.replace("\\", "/")).parts
            destination = (target / name).resolve()
            if (member.is_dir() or name.startswith(("/", "\\")) or
                    not parts or ".." in parts or
                    target.resolve() not in destination.parents or
                    not (name.endswith(".py") or name.endswith(".parquet") or
                         name.endswith("manifest.json"))):
                raise ValueError(f"Unsafe archive member: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as out:
                while block := source.read(1024 * 1024):
                    out.write(block)


def main() -> None:
    matches = list(Path("/kaggle/input").rglob("input_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one GPU probe input dataset, got {len(matches)}")
    source = matches[0].parent
    inputs = json.loads(matches[0].read_text(encoding="utf-8"))
    for name, expected in inputs["files_sha256"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Input SHA-256 mismatch: {name}")

    try:
        installed = importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    requirements = []
    if installed != "1.5.6":
        requirements.append("duckdb==1.5.6")
    try:
        importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        requirements.append("psutil")
    if requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise RuntimeError(f"Expected T4 on cuda:0, got {name}")

    project = Path("/kaggle/working/moviegraph_probe")
    project.mkdir(parents=True, exist_ok=False)
    extract_verified(source / "project_source.zip", project)
    (project / "cohorts").mkdir()
    summary = {"input_manifest": inputs, "cuda_device": "cuda:0", "gpu_name": name,
               "visible_gpus": torch.cuda.device_count(), "results": []}
    for cohort_name in inputs["cohorts"]:
        if "/" in cohort_name or "\\" in cohort_name or cohort_name in (".", ".."):
            raise ValueError("Invalid cohort name")
        extract_verified(source / f"{cohort_name}.zip", project / "cohorts",
                         {f"{cohort_name}/{name}" for name in
                          ("nodes.parquet", "edges.parquet", "manifest.json")})
        cohort = project / "cohorts" / cohort_name
        output = project / "results" / cohort_name
        output.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "scripts.gpu_feasibility_probe",
               "--cohort", str(cohort), "--out", str(output), "--device", "cuda:0",
               "--dim", "64", "--layers", "3", "--batch-size", "1024",
               "--steps", "5", "--eval-users", "100"]
        try:
            run = subprocess.run(cmd, cwd=project, capture_output=True, text=True,
                                 timeout=900)
            stdout, stderr, returncode = run.stdout, run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = None
        if not output.exists():
            output.mkdir()
        (output / "stdout.txt").write_text(stdout, encoding="utf-8")
        (output / "stderr.txt").write_text(stderr, encoding="utf-8")
        report_path = output / "report.json"
        record = {"cohort": cohort_name, "returncode": returncode,
                  "report_exists": report_path.is_file()}
        if returncode is None:
            record["status"] = "TIMEOUT"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record.update(status=report.get("status"),
                          graph=report.get("graph"),
                          gpu_peak_reserved_bytes=report.get("gpu_peak_reserved_bytes"),
                          gpu_total_headroom_fraction=report.get("gpu_total_headroom_fraction"),
                          gpu_effective_headroom_fraction=report.get("gpu_effective_headroom_fraction"))
        summary["results"].append(record)
        (project / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if returncode != 0 or (record.get("gpu_effective_headroom_fraction") is not None and
                              record["gpu_effective_headroom_fraction"] < .30):
            break


if __name__ == "__main__":
    main()
