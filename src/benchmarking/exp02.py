"""D13 EXP-02: normalization cadence R=1 versus R=5 on identical replay batches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import psutil

from src.benchmarking.common import ROOT, RssSampler, slice_events
from src.graph.full_builder import file_sha256
from src.graph.incremental import IncrementalGraph
from src.graph.normalize import ATOL, compare_adjacency
from src.graph.snapshot import (canonical_status, export_snapshot, freshness,
                                load_export, materialize_normalized)


FIELDS = ("run_id", "timestamp_utc", "dataset_id", "phase", "repetition", "refresh_every_batches",
          "batch_id", "canonical_version", "model_graph_version", "canonical_watermark_event_id",
          "model_watermark_event_id", "canonical_update_s", "normalization_s", "export_s",
          "cumulative_s", "app_bytes_written", "cumulative_export_bytes", "db_growth_bytes",
          "peak_rss_bytes", "rss_interval_ms", "os_read_bytes", "os_write_bytes",
          "staleness_batches", "staleness_events", "snapshot_count", "snapshot_nnz",
          "coo_sha256", "normalized_equivalence_pass", "normalized_max_abs_diff",
          "status", "failure_reason", "input_sha256", "batch_sha256", "git_commit",
          "git_dirty", "config_sha256", "code_sha256", "python_version", "duckdb_version",
          "cpu", "ram_bytes", "os", "cache_policy", "threads", "output_dir")


def prepare_batches(source: Path, output: Path, batch_size: int, events: int) -> list[Path]:
    if batch_size <= 0 or events <= 0 or events % batch_size:
        raise ValueError("Expected positive event count divisible by batch size")
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for start in range(0, events, batch_size):
        path = output / f"batch_{start // batch_size:04d}.parquet"
        slice_events(source, path, start, start + batch_size)
        paths.append(path)
    return paths


def worker(batch_paths: list[Path], output: Path, refresh_every: int,
           provenance: dict, run_id: str, phase: str, repetition: int) -> list[dict]:
    if refresh_every <= 0 or len(batch_paths) % refresh_every:
        raise ValueError("Refresh cadence must divide batch count")
    output.mkdir(parents=True, exist_ok=False)
    db_path = output / "state.duckdb"
    rows = []
    process = psutil.Process()
    with IncrementalGraph(db_path) as graph:
        baseline = materialize_normalized(graph, expected_version=0, provenance=provenance)
        if baseline["nnz"] != 0:
            raise AssertionError("Initial model graph must be empty")
        latest = 0
        cumulative = 0.0
        cumulative_bytes = 0
        snapshots = 0
        for batch_id, batch in enumerate(batch_paths):
            row = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "phase": phase, "repetition": repetition,
                   "refresh_every_batches": refresh_every, "batch_id": batch_id,
                   "batch_sha256": file_sha256(batch), "status": "PASS",
                   "normalization_s": 0.0, "export_s": 0.0, "app_bytes_written": 0}
            before_io = process.io_counters()
            before_db = db_path.stat().st_size
            with RssSampler() as sampler:
                start = time.perf_counter()
                applied = graph.apply_batch(batch_id, batch)
                row["canonical_update_s"] = time.perf_counter() - start
                if applied["status"] != "APPLIED" or applied["graph_version"] != batch_id + 1:
                    raise AssertionError("Unexpected canonical replay status/version")
                if (batch_id + 1) % refresh_every == 0:
                    start = time.perf_counter()
                    manifest = materialize_normalized(
                        graph, expected_version=batch_id + 1, provenance=provenance)
                    row["normalization_s"] = time.perf_counter() - start
                    latest = batch_id + 1
                    snapshots += 1
                    snapshot_dir = output / f"model_graph_v{latest}"
                    start = time.perf_counter()
                    export_snapshot(graph, latest, snapshot_dir)
                    row["export_s"] = time.perf_counter() - start
                    row["app_bytes_written"] = sum(path.stat().st_size for path in snapshot_dir.iterdir())
                    cumulative_bytes += row["app_bytes_written"]
                    row["snapshot_nnz"] = manifest["nnz"]
                    row["coo_sha256"] = manifest["coo_sha256"]
                status = freshness(graph, latest)
            after_io = process.io_counters()
            cumulative += row["canonical_update_s"] + row["normalization_s"] + row["export_s"]
            row.update(canonical_version=status["canonical_version"],
                       model_graph_version=status["model_graph_version"],
                       canonical_watermark_event_id=status["watermark_event_id"],
                       model_watermark_event_id=status["model_watermark_event_id"],
                       staleness_batches=status["staleness_batches"],
                       staleness_events=status["staleness_events"],
                       cumulative_s=cumulative, cumulative_export_bytes=cumulative_bytes,
                       snapshot_count=snapshots,
                       db_growth_bytes=max(0, db_path.stat().st_size-before_db),
                       peak_rss_bytes=sampler.peak, rss_interval_ms=int(sampler.interval*1000),
                       os_read_bytes=after_io.read_bytes-before_io.read_bytes,
                       os_write_bytes=after_io.write_bytes-before_io.write_bytes)
            rows.append(row)
            print(f"{run_id} {phase} rep={repetition} R={refresh_every} "
                  f"v={row['canonical_version']} update={row['canonical_update_s']:.3f}s "
                  f"normalize={row['normalization_s']:.3f}s export={row['export_s']:.3f}s",
                  flush=True)
    return rows


def worker_cli(args):
    paths = sorted(args.batches.glob("batch_*.parquet"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    try:
        rows = worker(paths, args.output, args.refresh_every, provenance,
                      args.run_id, args.phase, args.repetition)
    except Exception as exc:
        rows = [{"run_id": args.run_id, "phase": args.phase, "repetition": args.repetition,
                 "refresh_every_batches": args.refresh_every, "status": "ERROR",
                 "failure_reason": f"{type(exc).__name__}: {exc}"}]
    args.result.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


def compare_common(left_dir: Path, right_dir: Path, version: int) -> dict:
    left_manifest, left_nodes, left_adj = load_export(left_dir / f"model_graph_v{version}")
    right_manifest, right_nodes, right_adj = load_export(right_dir / f"model_graph_v{version}")
    comparison = compare_adjacency(left_adj, right_adj, atol=ATOL)
    comparison["pass"] = (comparison["pass"] and left_nodes == right_nodes and
                          left_manifest["coo_sha256"] == right_manifest["coo_sha256"])
    return comparison


def run(source: Path, output: Path, csv_path: Path, *, events=100_000,
        batch_size=10_000, repetitions=3, seed=7):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("EXP-02 output must be fresh")
    if csv_path.exists():
        raise FileExistsError("EXP-02 raw CSV must be fresh")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    output.mkdir(parents=True, exist_ok=True)
    batches = prepare_batches(source, output / "batches", batch_size, events)
    source_hash = file_sha256(source)
    config = {"events": events, "batch_size": batch_size, "refresh_every": [1, 5],
              "warmups": 1, "measured_repetitions": repetitions, "seed": seed,
              "threads": 1, "dtype": "float64", "atol": ATOL,
              "cache_policy": "warm-cache-uncontrolled", "rss_interval_ms": 5,
              "cost_boundary": "canonical apply + DB normalization materialization + Parquet export"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    code_hash = file_sha256(Path(__file__))
    machine = {"dataset_id": f"ml32m_chronological_{events}", "input_sha256": source_hash,
               "git_commit": git_commit, "git_dirty": git_dirty, "config_sha256": config_hash,
               "code_sha256": code_hash, "python_version": platform.python_version(),
               "duckdb_version": duckdb.__version__, "cpu": platform.processor(),
               "ram_bytes": psutil.virtual_memory().total, "os": platform.platform(),
               "cache_policy": config["cache_policy"], "threads": 1}
    provenance = {"dataset_id": machine["dataset_id"], "canonical_input_sha256": source_hash,
                  "config_sha256": config_hash, "code_sha256": code_hash}
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    rng = random.Random(seed)
    all_rows = []
    comparisons = []
    for phase, repetition in [("warmup", 0)] + [("measured", i) for i in range(1, repetitions+1)]:
        schedules = [1, 5]
        rng.shuffle(schedules)
        outputs = {}
        round_rows = []
        for refresh_every in schedules:
            target = output / f"{phase}_{repetition}_r{refresh_every}"
            result = output / f"{phase}_{repetition}_r{refresh_every}.json"
            command = [sys.executable, "-m", "src.benchmarking.exp02", "worker",
                       "--batches", str(output / "batches"), "--output", str(target),
                       "--result", str(result), "--provenance", str(provenance_path),
                       "--run-id", output.name, "--phase", phase,
                       "--repetition", str(repetition), "--refresh-every", str(refresh_every)]
            subprocess.run(command, cwd=ROOT, check=False)
            rows = json.loads(result.read_text(encoding="utf-8")) if result.exists() else [
                {"status": "ERROR", "failure_reason": "Worker returned without result file"}]
            for row in rows:
                row.update(machine)
                row["output_dir"] = str(target)
            round_rows.extend(rows)
            if len(rows) == len(batches) and all(row["status"] == "PASS" for row in rows):
                outputs[refresh_every] = target
        if set(outputs) == {1, 5}:
            for version in range(5, len(batches)+1, 5):
                comparison = compare_common(outputs[1], outputs[5], version)
                comparisons.append({"phase": phase, "repetition": repetition,
                                    "version": version, **comparison})
                for row in round_rows:
                    if row["canonical_version"] == version and row["normalization_s"] > 0:
                        row["normalized_equivalence_pass"] = comparison["pass"]
                        row["normalized_max_abs_diff"] = comparison["max_abs_diff"]
                        if not comparison["pass"]:
                            row["status"] = "CORRECTNESS_FAIL"
        all_rows.extend(round_rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    report = {"run_id": output.name, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "machine": machine, "rows": len(all_rows),
              "common_boundary_comparisons": comparisons,
              "status": "PASS" if (all(row["status"] == "PASS" for row in all_rows) and
                                  all(item["pass"] for item in comparisons) and
                                  len(comparisons) == (repetitions+1)*len(batches)//5) else "FAIL",
              "csv": str(csv_path)}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    work = sub.add_parser("worker")
    for name in ("batches", "output", "result", "provenance"):
        work.add_argument("--"+name, type=Path, required=True)
    work.add_argument("--run-id", required=True)
    work.add_argument("--phase", required=True)
    work.add_argument("--repetition", type=int, required=True)
    work.add_argument("--refresh-every", type=int, required=True)
    experiment = sub.add_parser("run")
    experiment.add_argument("--source", type=Path, default=ROOT / "data/processed/canonical_events/canonical_100000.parquet")
    experiment.add_argument("--out", type=Path, default=ROOT / "data/processed/d13_exp02" /
                            datetime.now(timezone.utc).strftime("D13-%Y%m%dT%H%M%SZ"))
    experiment.add_argument("--csv", type=Path, default=ROOT / "results/raw/exp02_normalization.csv")
    experiment.add_argument("--events", type=int, default=100_000)
    experiment.add_argument("--batch-size", type=int, default=10_000)
    experiment.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.command == "worker":
        raise SystemExit(worker_cli(args))
    print(json.dumps(run(args.source, args.out, args.csv, events=args.events,
                         batch_size=args.batch_size, repetitions=args.repetitions), indent=2))


if __name__ == "__main__":
    main()
