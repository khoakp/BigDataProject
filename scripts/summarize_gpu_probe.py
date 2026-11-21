"""Summarize measured phase reports without treating different GPUs as comparable."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ("report", "status", "gpu_name", "device", "nodes", "positive_edges",
          "cpu_peak_rss_gib", "gpu_peak_reserved_mib", "gpu_headroom_fraction",
          "load_s", "normalize_s", "sampler_s", "sparse_coo_s", "forward_s",
          "first_step_s", "remaining_steps_s", "ranking_s", "checkpoint_mib")


def row(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    phases = report.get("phases", {})
    def seconds(name):
        return phases.get(name, {}).get("wall_s")
    return {"report": str(path), "status": report.get("status"),
            "gpu_name": report["environment"].get("gpu_name"),
            "device": report["config"]["device"],
            "nodes": report.get("graph", {}).get("nodes"),
            "positive_edges": report.get("graph", {}).get("positive_edges"),
            "cpu_peak_rss_gib": (max((v["cpu_peak_rss_bytes"] for v in phases.values()),
                                     default=0) / 2**30),
            "gpu_peak_reserved_mib": report.get("gpu_peak_reserved_bytes", 0) / 2**20,
            "gpu_headroom_fraction": report.get("gpu_effective_headroom_fraction"),
            "load_s": seconds("load_python_graph"),
            "normalize_s": seconds("normalize_python"),
            "sampler_s": seconds("prepare_sampler"),
            "sparse_coo_s": seconds("to_sparse_coo"),
            "forward_s": seconds("forward"),
            "first_step_s": seconds("first_backward_adam_step"),
            "remaining_steps_s": seconds("short_trajectory"),
            "ranking_s": seconds("rank_train_catalog"),
            "checkpoint_mib": report.get("checkpoint_bytes", 0) / 2**20}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(row(path) for path in args.reports)


if __name__ == "__main__":
    main()
