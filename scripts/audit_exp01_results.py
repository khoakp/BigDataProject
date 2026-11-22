"""Audit preserved EXP-01 raw runs before reusing them for S06/S07.

Checks provenance, paired measurements, logical output hashes, measurement
denominators, exported files, and the published summary without rerunning the
benchmark or changing any raw artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.artifacts import file_sha256 as sha256_file

from src.benchmarking.exp01_analysis import summarize


TABLES = ("registry", "edges", "degrees")


def audit(raw: Path, summary_path: Path, report_path: Path, repo: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "PASS":
        raise ValueError("EXP-01 report is not PASS")
    rows, expected_summary = summarize(raw)
    if len(rows) != report["rows"]:
        raise ValueError("CSV row count differs from run report")
    if {row["run_id"] for row in rows} != {report["run_id"]}:
        raise ValueError("Run ID mismatch")
    machine = report["machine"]
    for csv_key, report_key in (("source_sha256", "source_sha256"),
                                ("git_commit", "git_commit"),
                                ("config_sha256", "config_sha256"),
                                ("cache_policy", "cache_policy"),
                                ("duckdb_version", "duckdb_version")):
        if {row[csv_key] for row in rows} != {str(machine[report_key])}:
            raise ValueError(f"Machine provenance mismatch: {csv_key}")
    if {int(row["threads"]) for row in rows} != {report["config"]["threads"]}:
        raise ValueError("Thread count differs from report")
    expected_cases = {tuple(case) for case in report["config"]["grid"]}
    if {(int(row["history_events"]), int(row["delta_events"])) for row in rows} != expected_cases:
        raise ValueError("CSV grid differs from report")
    if report["config"]["warmups"] != 1 or any(
            record["repetitions"] != report["config"]["measured_repetitions"]
            for record in expected_summary):
        raise ValueError("CSV warmup/repetition count differs from report")

    groups = defaultdict(dict)
    for row in rows:
        history, delta = int(row["history_events"]), int(row["delta_events"])
        method = row["method"]
        processed = history + delta if method == "full" else delta
        if int(row["events_processed"]) != processed:
            raise ValueError("Unexpected events_processed denominator")
        if not math.isclose(float(row["events_per_s"]),
                            processed / float(row["wall_s"]), rel_tol=1e-9):
            raise ValueError("events_per_s does not match wall time")
        directory = repo / Path(row["output_dir"].replace("\\", "/"))
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing export manifest: {directory}")
        export = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not {f"{table}_sha256" for table in TABLES} <= set(export):
            raise ValueError("Export manifest lacks table hashes")
        files = [directory / f"{table}.parquet" for table in TABLES]
        if any(not path.is_file() for path in files):
            raise ValueError(f"Missing Parquet export: {directory}")
        for table, path in zip(TABLES, files):
            if sha256_file(path) != export[f"{table}_sha256"]:
                raise ValueError(f"Parquet hash differs from manifest: {path}")
        size = manifest_path.stat().st_size + sum(path.stat().st_size for path in files)
        if int(row["export_bytes_written"]) != size:
            raise ValueError(f"Export byte count changed: {directory}")
        key = (history, delta, row["phase"], int(row["repetition"]))
        groups[key][method] = row
    for pair in groups.values():
        for field in ("registry_sha256", "edge_sha256", "degree_sha256"):
            if pair["full"][field] != pair["incremental"][field]:
                raise ValueError(f"Paired logical output differs: {field}")
        for table in TABLES:
            field = f"{table}_sha256"
            left_dir = repo / Path(pair["full"]["output_dir"].replace("\\", "/"))
            right_dir = repo / Path(pair["incremental"]["output_dir"].replace("\\", "/"))
            left = json.loads((left_dir / "manifest.json").read_text(encoding="utf-8"))
            right = json.loads((right_dir / "manifest.json").read_text(encoding="utf-8"))
            if left[field] != right[field]:
                raise ValueError(f"Paired Parquet output differs: {field}")

    with summary_path.open(newline="", encoding="utf-8") as stream:
        published = list(csv.DictReader(stream))
    by_case = {(int(row["history_events"]), int(row["delta_events"])): row
               for row in published}
    if len(by_case) != len(expected_summary):
        raise ValueError("Summary grid/row count differs from raw")
    for expected in expected_summary:
        case = (expected["history_events"], expected["delta_events"])
        if case not in by_case:
            raise ValueError(f"Missing summary case: {case}")
        actual = by_case[case]
        if set(actual) != set(expected):
            raise ValueError(f"Summary columns differ: {case}")
        for field, value in expected.items():
            if not math.isclose(float(actual[field]), float(value), rel_tol=1e-9,
                                abs_tol=1e-12):
                raise ValueError(f"Summary mismatch: {case} {field}")
    return {"run_id": report["run_id"], "status": "PASS", "rows": len(rows),
            "pairs": len(groups), "grid_cases": len(expected_cases),
            "measured_repetitions": report["config"]["measured_repetitions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.summary, args.report, args.repo)))


if __name__ == "__main__":
    main()
