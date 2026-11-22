"""Prepare and verify the frozen REC-01 input bundle for Kaggle GPU training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from src.artifacts import file_sha256 as sha256

import duckdb


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = "canonical_1000000.parquet"
SOURCE_SHA256 = "f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae"
EXPECTED = {
    "events": 1_000_000,
    "distinct_timestamps": 478_972,
    "train_end_ts": 840_100_781,
    "valid_end_ts": 841_267_862,
    "train_events": 802_988,
    "valid_events": 102_046,
    "test_events": 94_966,
}
REQUIRED_COLUMNS = {"event_id", "event_ts", "user_raw_id", "movie_raw_id",
                    "rating", "source_row_id"}


def inspect_source(source: Path, expected_hash: str = SOURCE_SHA256,
                   expected: dict = EXPECTED) -> dict:
    """Check content, canonical IDs, time cutoffs and three split event counts."""
    if sha256(source) != expected_hash:
        raise ValueError("Canonical Parquet SHA-256 differs from approved REC-01 source")
    with duckdb.connect() as con:
        columns = {row[0] for row in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(source)]).fetchall()}
        if not REQUIRED_COLUMNS.issubset(columns):
            raise ValueError("Canonical Parquet schema is incomplete")
        total, minimum, maximum, distinct = con.execute("""SELECT count(*),min(event_id),max(event_id),
            count(DISTINCT event_ts) FROM read_parquet(?)""", [str(source)]).fetchone()
        if (total, minimum, maximum, distinct) != (
                expected["events"], 0, expected["events"] - 1, expected["distinct_timestamps"]):
            raise ValueError("Canonical event count, ID range or timestamp count differs")
        train_idx = min(max(0, int(distinct * .8) - 1), distinct - 3)
        valid_idx = min(max(train_idx + 1, int(distinct * .9) - 1), distinct - 2)
        ordered_times = "SELECT DISTINCT event_ts FROM read_parquet(?) ORDER BY event_ts"
        train_end = con.execute(f"SELECT event_ts FROM ({ordered_times}) LIMIT 1 OFFSET ?",
                                [str(source), train_idx]).fetchone()[0]
        valid_end = con.execute(f"SELECT event_ts FROM ({ordered_times}) LIMIT 1 OFFSET ?",
                                [str(source), valid_idx]).fetchone()[0]
        train_count, valid_count, test_count = con.execute("""SELECT
            count(*) FILTER (WHERE event_ts <= ?),
            count(*) FILTER (WHERE event_ts > ? AND event_ts <= ?),
            count(*) FILTER (WHERE event_ts > ?)
            FROM read_parquet(?)""", [train_end, train_end, valid_end, valid_end, str(source)]).fetchone()
    observed = {"events": total, "distinct_timestamps": distinct,
                "train_end_ts": train_end, "valid_end_ts": valid_end,
                "train_events": train_count, "valid_events": valid_count, "test_events": test_count}
    if observed != expected:
        raise ValueError(f"Canonical temporal protocol differs: {observed}")
    return observed


def source_files(root: Path) -> list[Path]:
    files = sorted((root / "src").rglob("*.py"))
    files.append(root / "scripts" / "rec01_kaggle_prep.py")
    required = ("src/models/rec01_train.py", "src/evaluation/rec01.py",
                "src/evaluation/baselines.py", "src/evaluation/checkpoint_selection.py")
    if any(not (root / rel).is_file() for rel in required):
        raise FileNotFoundError("REC-01 training/evaluation source is incomplete")
    return files


def write_source_zip(root: Path, target: Path) -> dict[str, str]:
    """Zip only Python source with deterministic paths, order and timestamps."""
    hashes = {}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in source_files(root):
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            hashes[relative] = hashlib.sha256(payload).hexdigest()
    return hashes


def prepare(source: Path, output: Path, root: Path = ROOT) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Bundle output must be a fresh empty directory")
    observed = inspect_source(source)
    output.mkdir(parents=True, exist_ok=True)
    copied = output / SOURCE_NAME
    shutil.copyfile(source, copied)
    if sha256(copied) != SOURCE_SHA256:
        raise AssertionError("Copied canonical data hash changed")
    archive = output / "rec01_source.zip"
    code_hashes = write_source_zip(root, archive)
    for relative, expected_hash in code_hashes.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        if sha256(destination) != expected_hash:
            raise AssertionError(f"Copied source hash changed: {relative}")
    original_manifest = root / "data/processed/canonical_events/manifest.json"
    if original_manifest.is_file():
        d02_data = json.loads(original_manifest.read_text(encoding="utf-8"))
        prefix = d02_data.get("prefixes", {}).get("1000000", {})
        if prefix.get("sha256") != SOURCE_SHA256 or prefix.get("rows") != EXPECTED["events"]:
            raise ValueError("D02 manifest does not match the approved REC-01 canonical prefix")
        shutil.copyfile(original_manifest, output / "d02_manifest.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    manifest = {"dataset_id": "ml32m_chronological_1000000_rec01",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": SOURCE_SHA256, "source_bytes": copied.stat().st_size,
                "approved_protocol": observed, "code_sha256": code_hashes,
                "archive_sha256": sha256(archive), "git_head": commit,
                "git_dirty": dirty, "local_python": platform.python_version(),
                "local_duckdb": duckdb.__version__,
                "d02_manifest_sha256": sha256(output / "d02_manifest.json")
                if (output / "d02_manifest.json").is_file() else None}
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    verify_input(output)
    return manifest


def verify_input(bundle: Path) -> dict:
    manifest = json.loads((bundle / "input_manifest.json").read_text(encoding="utf-8"))
    source = bundle / SOURCE_NAME
    if manifest["source_sha256"] != SOURCE_SHA256 or sha256(source) != SOURCE_SHA256:
        raise ValueError("Bundle canonical source hash mismatch")
    if source.stat().st_size != manifest["source_bytes"]:
        raise ValueError("Bundle canonical source size mismatch")
    if inspect_source(source) != manifest["approved_protocol"]:
        raise ValueError("Bundle temporal protocol mismatch")
    archive = bundle / "rec01_source.zip"
    if archive.is_file():
        if sha256(archive) != manifest["archive_sha256"]:
            raise ValueError("Bundle source archive hash mismatch")
        with zipfile.ZipFile(archive) as stream:
            if set(stream.namelist()) != set(manifest["code_sha256"]):
                raise ValueError("Bundle source file list mismatch")
            for name, expected_hash in manifest["code_sha256"].items():
                if hashlib.sha256(stream.read(name)).hexdigest() != expected_hash:
                    raise ValueError(f"Bundle source archive content mismatch: {name}")
    for name, expected_hash in manifest["code_sha256"].items():
        if sha256(bundle / name) != expected_hash:
            raise ValueError(f"Bundle source content mismatch: {name}")
    d02 = bundle / "d02_manifest.json"
    if manifest["d02_manifest_sha256"] is not None and sha256(d02) != manifest["d02_manifest_sha256"]:
        raise ValueError("D02 source manifest hash mismatch")
    return manifest


def _verify_run_core(bundle_manifest: dict, run: Path) -> dict:
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("REC-01 run report is not PASS")
    source = report.get("source", {})
    if source.get("sha256") != bundle_manifest["source_sha256"]:
        raise ValueError("REC-01 run used a different canonical source")
    for field, value in EXPECTED.items():
        if report.get("protocol", {}).get(field) != value:
            raise ValueError(f"REC-01 run protocol differs: {field}")
    config_hash = hashlib.sha256(json.dumps(report["protocol"], sort_keys=True).encode()).hexdigest()
    if config_hash != report.get("config_sha256"):
        raise ValueError("REC-01 run config hash mismatch")
    if not report.get("graph", {}).get("full_equivalence", {}).get("pass"):
        raise ValueError("REC-01 graph equivalence did not pass")
    graph_export = run / "model_graph"
    if (not (graph_export / "nodes.parquet").is_file() or
            not (graph_export / "adjacency.parquet").is_file() or
            sha256(graph_export / "manifest.json") != report["graph"].get("model_graph_manifest_sha256")):
        raise ValueError("REC-01 model graph export missing or changed")
    from src.graph.snapshot import load_export
    load_export(graph_export)
    code_hashes = report.get("code_sha256", {})
    required_code = {"src/models/rec01_train.py", "src/models/lightgcn.py",
                     "src/graph/snapshot.py", "src/evaluation/rec01.py"}
    if not required_code.issubset(code_hashes):
        raise ValueError("REC-01 run code provenance is incomplete")
    for name, expected_hash in code_hashes.items():
        if bundle_manifest["code_sha256"].get(name) != expected_hash:
            raise ValueError(f"REC-01 run code differs from uploaded bundle: {name}")
    steps = report["protocol"]["checkpoint_steps"]
    candidates = report.get("validation_candidates", [])
    if [item["step"] for item in candidates] != steps:
        raise ValueError("REC-01 validation checkpoint schedule incomplete")
    for item in candidates:
        checkpoint = run / f"model_step_{item['step']:04d}.pt"
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"REC-01 checkpoint hash mismatch: {checkpoint.name}")
    selected = report.get("selected_checkpoint", {})
    if selected not in candidates or not report.get("selected_checkpoint_reloaded"):
        raise ValueError("REC-01 selected checkpoint was not validated/reloaded")
    if report.get("selected_checkpoint_file") != f"model_step_{selected['step']:04d}.pt":
        raise ValueError("REC-01 selected checkpoint file mismatch")
    if sha256(run / "model_ids.json") != report.get("model_ids_sha256"):
        raise ValueError("REC-01 ID mapping hash mismatch")
    for split in ("validation", "test"):
        cohort = report.get("cohorts", {}).get(split, {})
        metrics = report.get("metrics", {}).get(split, {})
        if not cohort.get("warm_users"):
            raise ValueError(f"REC-01 {split} warm cohort is empty")
        for method in ("lightgcn", "popularity"):
            result = metrics.get(method, {})
            if result.get("users") != cohort["warm_users"]:
                raise ValueError(f"REC-01 {split} methods used different cohorts")
            for field in ("recall", "ndcg"):
                value = result.get(field)
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"REC-01 {split} {method} {field} invalid")
    with (run / "rec01_quality.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(steps) + 3 or any(row["status"] != "PASS" for row in rows):
        raise ValueError("REC-01 raw CSV is incomplete")
    if any(row["source_sha256"] != SOURCE_SHA256 or row["config_sha256"] != config_hash
           for row in rows):
        raise ValueError("REC-01 raw CSV provenance mismatch")
    if not (run / "rankings.json").is_file() or not (run / "graph_build" / "graph.duckdb").is_file():
        raise ValueError("REC-01 ranking or graph artifact missing")
    return report


def seal_output(bundle: Path, run: Path) -> dict:
    """Write hashes for every output artifact after a completed Kaggle run."""
    bundle_manifest = verify_input(bundle)
    report = _verify_run_core(bundle_manifest, run)
    target = run / "output_manifest.json"
    if target.exists():
        raise FileExistsError("REC-01 output manifest already exists")
    hashes = {path.relative_to(run).as_posix(): sha256(path) for path in sorted(run.rglob("*"))
              if path.is_file()}
    manifest = {"run_id": report["run_id"], "sealed_utc": datetime.now(timezone.utc).isoformat(),
                "input_source_sha256": bundle_manifest["source_sha256"],
                "input_archive_sha256": bundle_manifest["archive_sha256"],
                "files_sha256": hashes}
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_output(bundle: Path, run: Path) -> dict:
    """Verify a downloaded Kaggle output tree against its sealed file list."""
    bundle_manifest = verify_input(bundle)
    report = _verify_run_core(bundle_manifest, run)
    manifest = json.loads((run / "output_manifest.json").read_text(encoding="utf-8"))
    if (manifest["run_id"] != report["run_id"] or
            manifest["input_source_sha256"] != bundle_manifest["source_sha256"] or
            manifest["input_archive_sha256"] != bundle_manifest["archive_sha256"]):
        raise ValueError("REC-01 output manifest provenance mismatch")
    observed = {path.relative_to(run).as_posix() for path in run.rglob("*")
                if path.is_file() and path.name != "output_manifest.json"}
    if observed != set(manifest["files_sha256"]):
        raise ValueError("REC-01 output file list differs from sealed manifest")
    for relative, expected_hash in manifest["files_sha256"].items():
        if sha256(run / relative) != expected_hash:
            raise ValueError(f"REC-01 downloaded file hash mismatch: {relative}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path,
                      default=ROOT / "data/processed/canonical_events" / SOURCE_NAME)
    prep.add_argument("--out", type=Path,
                      default=ROOT / "data/processed/rec01_kaggle_bundle")
    check = sub.add_parser("verify-input")
    check.add_argument("--bundle", type=Path, required=True)
    seal = sub.add_parser("seal-output")
    seal.add_argument("--bundle", type=Path, required=True)
    seal.add_argument("--run", type=Path, required=True)
    verify = sub.add_parser("verify-output")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.out)
    elif args.command == "verify-input":
        result = verify_input(args.bundle)
    elif args.command == "seal-output":
        result = seal_output(args.bundle, args.run)
    else:
        result = verify_output(args.bundle, args.run)
    print(json.dumps({"status": "PASS", "result": result.get("run_id", result.get("dataset_id")),
                      "source_sha256": result.get("source_sha256", result.get("input_source_sha256"))},
                     indent=2))


if __name__ == "__main__":
    main()
