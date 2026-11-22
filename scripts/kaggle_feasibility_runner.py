"""Kaggle T4 script: verify inputs, then probe one GPU (cuda:0) by cohort.

The Kaggle metadata attaches a private dataset made by prepare_kaggle_feasibility.
Each cohort runs in its own Python process so CPU/GPU peaks are independent.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


# Kaggle CLI uploads only kernel-metadata.json's code_file. A one-off trajectory
# may be embedded in the submitted copy of this script; the repository default
# remains the 5-step sweep over all packaged cohorts.
EMBEDDED_PROBE_PLAN = None


def probe_status(returncode: int | None, report_status: str | None) -> str:
    if returncode is None:
        return "TIMEOUT"
    if returncode != 0 and report_status in (None, "RUNNING", "PASS"):
        return "ERROR"
    return report_status or "ERROR"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_mounted_files(source: Path, inputs: dict) -> tuple[dict[str, Path], list[str]]:
    """Verify Kaggle's auto-extracted ZIP members and any retained ZIP files."""
    root = source.resolve()
    members = inputs["unpacked_files_sha256"]
    if not members or "project_source/scripts/gpu_feasibility_probe.py" not in members:
        raise ValueError("Missing unpacked project source manifest")
    for cohort in inputs["cohorts"]:
        if "/" in cohort or "\\" in cohort or cohort in (".", ".."):
            raise ValueError("Invalid cohort name")
        for name in ("nodes.parquet", "edges.parquet", "manifest.json"):
            if f"{cohort}/{cohort}/{name}" not in members:
                raise ValueError(f"Missing unpacked cohort manifest: {cohort}/{name}")
    verified = {}
    for relative, expected in members.items():
        parts = PurePosixPath(relative).parts
        if (not parts or relative.startswith(("/", "\\")) or ".." in parts or
                "\\" in relative):
            raise ValueError(f"Unsafe input manifest path: {relative}")
        path = source.joinpath(*parts)
        resolved = path.resolve()
        if root not in resolved.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe mounted input: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Unpacked SHA-256 mismatch: {relative}")
        verified[relative] = path
    retained_archives = []
    for name, expected in inputs["files_sha256"].items():
        if name != Path(name).name or not name.endswith(".zip"):
            raise ValueError(f"Unsafe archive name: {name}")
        path = source / name
        if path.is_file():
            if sha256(path) != expected:
                raise ValueError(f"ZIP SHA-256 mismatch: {name}")
            retained_archives.append(name)
        elif path.exists():
            raise ValueError(f"Unexpected archive path: {name}")
    return verified, retained_archives


def load_probe_plan(path: Path | None, available_cohorts: list[str],
                    embedded: dict | None = None) -> dict:
    """Use the original five-step sweep unless a bounded plan is supplied."""
    if embedded is not None:
        plan, source = embedded, "embedded"
    elif path is not None and path.is_file():
        plan = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    else:
        return {"cohorts": list(available_cohorts), "steps": 5, "source": "default"}
    if not isinstance(plan, dict) or set(plan) != {"cohorts", "steps"}:
        raise ValueError("probe_plan.json needs exactly cohorts and steps")
    selected = plan["cohorts"]
    steps = plan["steps"]
    if (not isinstance(selected, list) or not selected or
            any(not isinstance(name, str) or name not in available_cohorts
                for name in selected) or len(set(selected)) != len(selected)):
        raise ValueError("probe_plan.json cohorts must be a unique nonempty subset")
    if type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError("probe_plan.json steps must be an integer in 1..500")
    return {"cohorts": selected, "steps": steps, "source": source}


def main() -> None:
    matches = list(Path("/kaggle/input").rglob("input_manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one GPU probe input dataset, got {len(matches)}")
    source = matches[0].parent
    inputs = json.loads(matches[0].read_text(encoding="utf-8"))
    verified, retained_archives = verify_mounted_files(source, inputs)
    plan_path = next((path for path in
                      (Path(__file__).resolve().with_name("probe_plan.json"),
                       Path.cwd() / "probe_plan.json") if path.is_file()), None)
    plan = load_probe_plan(plan_path, inputs["cohorts"], EMBEDDED_PROBE_PLAN)

    try:
        installed = importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    requirements = []
    if installed is None:
        requirements.append("duckdb==1.5.6")
    try:
        importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        requirements.append("psutil")
    if requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)
    if installed is None:
        installed = importlib.metadata.version("duckdb")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise RuntimeError(f"Expected T4 on cuda:0, got {name}")

    project = Path("/kaggle/working/moviegraph_probe")
    project.mkdir(parents=True, exist_ok=False)
    for relative, path in verified.items():
        if not relative.startswith("project_source/"):
            continue
        target = project / relative.removeprefix("project_source/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    summary = {"input_manifest": inputs, "cuda_device": "cuda:0", "gpu_name": name,
               "visible_gpus": torch.cuda.device_count(),
               "duckdb_version": installed,
               "retained_archives_verified": retained_archives,
               "probe_plan": plan,
               "results": []}
    for cohort_name in plan["cohorts"]:
        cohort = source / cohort_name / cohort_name
        output = project / "results" / cohort_name
        output.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "scripts.gpu_feasibility_probe",
               "--cohort", str(cohort), "--out", str(output), "--device", "cuda:0",
               "--dim", "64", "--layers", "3", "--batch-size", "1024",
               "--steps", str(plan["steps"]), "--eval-users", "100"]
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
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record.update(status=report.get("status"),
                          graph=report.get("graph"),
                          cpu_effective_available_min_bytes=report.get("cpu_effective_available_min_bytes"),
                          cpu_effective_headroom_fraction=report.get("cpu_effective_headroom_fraction"),
                          gpu_peak_reserved_bytes=report.get("gpu_peak_reserved_bytes"),
                          gpu_total_headroom_fraction=report.get("gpu_total_headroom_fraction"),
                          gpu_effective_headroom_fraction=report.get("gpu_effective_headroom_fraction"))
        record["status"] = probe_status(returncode, record.get("status"))
        summary["results"].append(record)
        (project / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if (returncode != 0 or
                (record.get("gpu_effective_headroom_fraction") is not None and
                 record["gpu_effective_headroom_fraction"] < .30) or
                (record.get("cpu_effective_available_min_bytes") is not None and
                 record["cpu_effective_available_min_bytes"] < 2 * 1024**3)):
            break


if __name__ == "__main__":
    main()
