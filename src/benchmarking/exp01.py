"""D11 EXP-01 pilot: fair Full Rebuild versus stateful Incremental update."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import psutil

from src.graph.equivalence import compare_states, full_state
from src.graph.full_builder import file_sha256, rebuild
from src.graph.incremental import IncrementalGraph
from src.ingest.d02_profile import run as canonicalize
from src.benchmarking.common import ROOT, RssSampler, slice_events, sql_path


TABLES = ("registry", "edges", "degrees")
FIELDS = ("run_id", "timestamp_utc", "history_events", "delta_events", "method", "phase",
          "repetition", "status", "failure_reason", "wall_s", "cpu_s", "peak_rss_bytes",
          "rss_interval_ms", "app_bytes_read", "export_bytes_written", "db_growth_bytes",
          "os_read_bytes", "os_write_bytes", "events_processed", "events_per_s",
          "positive_edges_inserted", "registry_sha256", "edge_sha256", "degree_sha256",
          "correctness_pass", "prebuild_wall_s", "prebuild_cpu_s", "prebuild_db_bytes",
          "cache_policy", "threads", "input_sha256", "history_sha256", "delta_sha256",
          "git_commit", "git_dirty", "config_sha256", "python_version", "duckdb_version",
          "cpu", "ram_bytes", "os", "source_sha256", "output_dir")


def export_incremental(graph: IncrementalGraph, output: Path) -> dict:
    """Match D03's three logical Parquet tables and column/order contract."""
    output.mkdir(parents=True, exist_ok=True)
    queries = {
        "registry": "SELECT node_id,node_type,raw_id,first_event_id FROM node_registry ORDER BY node_id",
        "edges": """SELECT e.user_node_id,e.movie_node_id,u.raw_id AS user_raw_id,
                    m.raw_id AS movie_raw_id,e.first_event_id FROM edge_state e
                    JOIN node_registry u ON u.node_id=e.user_node_id
                    JOIN node_registry m ON m.node_id=e.movie_node_id
                    ORDER BY u.raw_id,m.raw_id""",
        "degrees": "SELECT node_id,degree FROM degree_state ORDER BY node_id",
    }
    hashes = {}
    for name, query in queries.items():
        graph.con.execute(f"COPY ({query}) TO '{sql_path(output / (name + '.parquet'))}' (FORMAT PARQUET)")
        hashes[name + "_sha256"] = file_sha256(output / (name + ".parquet"))
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return hashes


def worker(method: str, combined: Path, delta: Path, prebuilt_db: Path, output: Path) -> dict:
    """Measure update+same export. Copy the prebuilt DB before the timer."""
    if method not in ("full", "incremental"):
        raise ValueError("Unknown method")
    output.mkdir(parents=True, exist_ok=False)
    db_path = output / "working.duckdb"
    db_before = 0
    if method == "incremental":
        shutil.copy2(prebuilt_db, db_path)
        db_before = db_path.stat().st_size
    process = psutil.Process()
    before_io = process.io_counters()
    with RssSampler() as sampler:
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        if method == "full":
            result = rebuild(combined, output)
            inserted = result["edges"]
            processed = result["events"]
        else:
            with IncrementalGraph(db_path) as graph:
                result = graph.apply_batch(1, delta)
                export_incremental(graph, output)
            inserted = result["inserted_edges"]
            with duckdb.connect() as con:
                processed = con.execute("SELECT count(*) FROM read_parquet(?)", [str(delta)]).fetchone()[0]
        wall = time.perf_counter() - start_wall
        cpu = time.process_time() - start_cpu
    after_io = process.io_counters()
    export_bytes = sum((output / (name + ".parquet")).stat().st_size for name in TABLES)
    export_bytes += (output / "manifest.json").stat().st_size
    return {"wall_s": wall, "cpu_s": cpu, "peak_rss_bytes": sampler.peak,
            "rss_interval_ms": int(sampler.interval * 1000),
            "app_bytes_read": (combined if method == "full" else delta).stat().st_size,
            "export_bytes_written": export_bytes,
            "db_growth_bytes": max(0, db_path.stat().st_size - db_before) if method == "incremental" else 0,
            "os_read_bytes": after_io.read_bytes - before_io.read_bytes,
            "os_write_bytes": after_io.write_bytes - before_io.write_bytes,
            "events_processed": processed, "events_per_s": processed / wall,
            "positive_edges_inserted": inserted,
            "input_sha256": file_sha256(combined if method == "full" else delta)}


def run_worker(args):
    try:
        result = worker(args.method, args.combined, args.delta, args.prebuilt_db, args.output)
        result["status"] = "PASS"
    except Exception as exc:
        result = {"status": "ERROR", "failure_reason": f"{type(exc).__name__}: {exc}"}
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


def prepare_history(source: Path, root: Path, history: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    history_path = root / "history.parquet"
    slice_events(source, history_path, 0, history)
    db_path = root / "prebuilt.duckdb"
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    with IncrementalGraph(db_path) as graph:
        graph.apply_batch(0, history_path)
    return {"history_path": history_path, "db_path": db_path,
            "prebuild_wall_s": time.perf_counter() - start_wall,
            "prebuild_cpu_s": time.process_time() - start_cpu,
            "prebuild_db_bytes": db_path.stat().st_size,
            "history_sha256": file_sha256(history_path)}


def case_inputs(source: Path, root: Path, history: int, delta: int):
    root.mkdir(parents=True, exist_ok=True)
    combined, delta_path = root / "combined.parquet", root / "delta.parquet"
    slice_events(source, combined, 0, history + delta)
    slice_events(source, delta_path, history, history + delta)
    return combined, delta_path


def run_pilot(source: Path, output: Path, csv_path: Path, grid, *, raw_ratings=None, seed=7,
              repetitions=1):
    """One warmup plus independent measured processes per method/case."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Pilot output must be a fresh directory")
    if csv_path.exists():
        raise FileExistsError("Raw CSV must be a fresh file")
    output.mkdir(parents=True, exist_ok=True)
    sizes = sorted({h for h, _ in grid})
    max_events = max(h + d for h, d in grid)
    if raw_ratings is not None:
        canonical_dir = output / "canonical"
        canonicalize(raw_ratings, canonical_dir, max_prefix=max_events)
        source = canonical_dir / f"canonical_{max_events}.parquet"
    with duckdb.connect() as con:
        count = con.execute("SELECT count(*) FROM read_parquet(?)", [str(source)]).fetchone()[0]
    if count < max_events:
        raise ValueError("Canonical source is shorter than pilot grid")
    run_id = output.name
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    config = {"grid": grid, "seed": seed, "warmups": 1, "measured_repetitions": repetitions,
              "threads": 1, "cache_policy": "warm-cache-uncontrolled", "rss_interval_ms": 5,
              "measurement_boundary": "update or full rebuild plus 3-table Parquet export"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    machine = {"git_commit": git_head, "git_dirty": git_dirty, "config_sha256": config_hash,
               "python_version": platform.python_version(), "duckdb_version": duckdb.__version__,
               "cpu": platform.processor(), "ram_bytes": psutil.virtual_memory().total,
               "os": platform.platform(), "cache_policy": config["cache_policy"],
               "threads": 1, "source_sha256": file_sha256(source)}
    histories = {h: prepare_history(source, output / f"h{h}", h) for h in sizes}
    rng = random.Random(seed)
    rows = []
    for history, delta in grid:
        case_dir = output / f"h{history}_d{delta}"
        combined, delta_path = case_inputs(source, case_dir, history, delta)
        outcomes = {}
        rounds = [("warmup", 0)] + [("measured", i) for i in range(1, repetitions + 1)]
        for phase, repetition in rounds:
            order = ["full", "incremental"]
            rng.shuffle(order)
            for method in order:
                target = case_dir / f"{phase}_{repetition}_{method}"
                result_file = case_dir / f"{phase}_{repetition}_{method}.json"
                command = [sys.executable, "-m", "src.benchmarking.exp01", "worker",
                           "--method", method, "--combined", str(combined), "--delta", str(delta_path),
                           "--prebuilt-db", str(histories[history]["db_path"]),
                           "--output", str(target), "--result", str(result_file)]
                subprocess.run(command, cwd=ROOT, check=False)
                result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {
                    "status": "ERROR", "failure_reason": "Worker returned without result file"}
                row = {**machine, **{k: histories[history][k] for k in (
                    "prebuild_wall_s", "prebuild_cpu_s", "prebuild_db_bytes", "history_sha256")},
                    **result, "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "history_events": history, "delta_events": delta, "method": method, "phase": phase,
                    "repetition": repetition, "output_dir": str(target),
                    "delta_sha256": file_sha256(delta_path)}
                rows.append(row)
                if result["status"] == "PASS":
                    outcomes[(phase, repetition, method)] = target
        for phase, repetition in rounds:
            if (phase, repetition, "full") in outcomes and (phase, repetition, "incremental") in outcomes:
                left = full_state(outcomes[(phase, repetition, "full")])
                right = full_state(outcomes[(phase, repetition, "incremental")])
                comparison = compare_states(left, right)
                for row in rows:
                    if (row["history_events"] == history and row["delta_events"] == delta and
                            row["phase"] == phase and row["repetition"] == repetition):
                        side = "full" if row["method"] == "full" else "incremental"
                        row.update(correctness_pass=comparison["pass"],
                                   registry_sha256=comparison["registry"][side + "_sha256"],
                                   edge_sha256=comparison["edges"][side + "_sha256"],
                                   degree_sha256=comparison["degrees"][side + "_sha256"])
                        if not comparison["pass"]:
                            row["status"] = "CORRECTNESS_FAIL"
            else:
                for row in rows:
                    if (row["history_events"] == history and row["delta_events"] == delta and
                            row["phase"] == phase and row["repetition"] == repetition):
                        row["correctness_pass"] = False
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    manifest = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "machine": machine, "rows": len(rows),
                "status": "PASS" if all(r["status"] == "PASS" and r["correctness_pass"] for r in rows) else "FAIL",
                "csv": str(csv_path)}
    (output / "report.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_grid(value: str):
    pairs = []
    for item in value.split(","):
        h, d = item.split(":")
        pairs.append((int(h), int(d)))
    if not pairs or any(h <= 0 or d <= 0 for h, d in pairs):
        raise ValueError("Grid must contain positive H:D pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    work = sub.add_parser("worker")
    for name in ("combined", "delta", "prebuilt-db", "output", "result"):
        work.add_argument("--" + name, type=Path, required=True)
    work.add_argument("--method", choices=("full", "incremental"), required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--source", type=Path, default=ROOT / "data/processed/canonical_events/canonical_1000000.parquet")
    pilot.add_argument("--ratings", type=Path, help="Regenerate canonical prefix from raw ratings.csv")
    pilot.add_argument("--out", type=Path, default=ROOT / "data/processed/d11_exp01" /
                       datetime.now(timezone.utc).strftime("D11-%Y%m%dT%H%M%SZ"))
    pilot.add_argument("--csv", type=Path, default=ROOT / "results/raw/exp01_pilot_d11.csv")
    pilot.add_argument("--grid", type=parse_grid, default=parse_grid("100000:1000,100000:10000,500000:1000,500000:10000,1000000:1000,1000000:10000"))
    repeat = sub.add_parser("repeat")
    repeat.add_argument("--source", type=Path, default=ROOT / "data/processed/d11_exp01/D11-pilot-grid/canonical/canonical_1010000.parquet")
    repeat.add_argument("--ratings", type=Path)
    repeat.add_argument("--out", type=Path, default=ROOT / "data/processed/d12_exp01" /
                        datetime.now(timezone.utc).strftime("D12-%Y%m%dT%H%M%SZ"))
    repeat.add_argument("--csv", type=Path, default=ROOT / "results/raw/exp01_full_vs_incremental.csv")
    repeat.add_argument("--grid", type=parse_grid, default=parse_grid("100000:1000,100000:10000,500000:1000,500000:10000,1000000:1000,1000000:10000"))
    repeat.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.command == "worker":
        raise SystemExit(run_worker(args))
    print(json.dumps(run_pilot(args.source, args.out, args.csv, args.grid,
                               raw_ratings=args.ratings,
                               repetitions=getattr(args, "repetitions", 1)), indent=2))


if __name__ == "__main__":
    main()
