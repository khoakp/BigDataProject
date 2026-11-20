"""Summarize and plot D13 EXP-02 from its measured raw CSV."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def summarize(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 80 or any(row["status"] != "PASS" for row in rows):
        raise ValueError("Expected 80 PASS rows: warmup plus three repetitions, R=1/5, ten batches")
    measured = [row for row in rows if row["phase"] == "measured"]
    if len(measured) != 60:
        raise ValueError("Expected 60 measured rows")
    groups = defaultdict(list)
    for row in measured:
        groups[(int(row["refresh_every_batches"]), int(row["repetition"]))].append(row)
    if set(groups) != {(r, rep) for r in (1, 5) for rep in (1, 2, 3)}:
        raise ValueError("Missing regime/repetition")
    runs = []
    for (refresh, repetition), batch_rows in sorted(groups.items()):
        batch_rows.sort(key=lambda row: int(row["canonical_version"]))
        if [int(row["canonical_version"]) for row in batch_rows] != list(range(1, 11)):
            raise ValueError("Incomplete canonical version sequence")
        boundaries = [row for row in batch_rows if row["normalized_equivalence_pass"]]
        if len(boundaries) != 2 or any(row["normalized_equivalence_pass"] != "True" for row in boundaries):
            raise ValueError("Common-boundary comparison missing or failed")
        runs.append({"refresh_every_batches": refresh, "repetition": repetition,
                     "total_s": float(batch_rows[-1]["cumulative_s"]),
                     "canonical_update_s": sum(float(row["canonical_update_s"]) for row in batch_rows),
                     "normalization_s": sum(float(row["normalization_s"]) for row in batch_rows),
                     "export_s": sum(float(row["export_s"]) for row in batch_rows),
                     "export_bytes": int(batch_rows[-1]["cumulative_export_bytes"]),
                     "materializations": int(batch_rows[-1]["snapshot_count"]),
                     "peak_sampled_rss_bytes": max(int(row["peak_rss_bytes"]) for row in batch_rows),
                     "max_staleness_batches": max(int(row["staleness_batches"]) for row in batch_rows),
                     "mean_staleness_batches": stats.mean(int(row["staleness_batches"]) for row in batch_rows),
                     "max_staleness_events": max(int(row["staleness_events"]) for row in batch_rows)})
    regime = {}
    metrics = ("total_s", "canonical_update_s", "normalization_s", "export_s",
               "export_bytes", "materializations", "peak_sampled_rss_bytes",
               "max_staleness_batches", "mean_staleness_batches", "max_staleness_events")
    for refresh in (1, 5):
        sample = [run for run in runs if run["refresh_every_batches"] == refresh]
        regime[str(refresh)] = {key: {"median": stats.median(run[key] for run in sample),
                                       "min": min(run[key] for run in sample),
                                       "max": max(run[key] for run in sample)} for key in metrics}
    return {"raw_csv": str(csv_path), "measured_runs": runs, "regime": regime,
            "total_wall_ratio_r1_over_r5": regime["1"]["total_s"]["median"] /
                                            regime["5"]["total_s"]["median"]}


def plot(csv_path: Path, asset_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["phase"] == "measured"]
    asset_dir.mkdir(parents=True, exist_ok=True)
    colors = {1: "#1f77b4", 5: "#d95f02"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    for refresh in (1, 5):
        regime_rows = [row for row in rows if int(row["refresh_every_batches"]) == refresh]
        versions = list(range(1, 11))
        cumulative = [stats.median(float(row["cumulative_s"]) for row in regime_rows
                                   if int(row["canonical_version"]) == v) for v in versions]
        export_mb = [stats.median(int(row["cumulative_export_bytes"]) for row in regime_rows
                                  if int(row["canonical_version"]) == v) / 1e6 for v in versions]
        lag = [stats.median(int(row["staleness_batches"]) for row in regime_rows
                            if int(row["canonical_version"]) == v) for v in versions]
        axes[0].plot(versions, cumulative, marker="o", color=colors[refresh], label=f"R={refresh}")
        axes[1].step(versions, lag, where="mid", marker="o", color=colors[refresh], label=f"R={refresh}")
        axes[2].step(versions, export_mb, where="post", marker="o", color=colors[refresh], label=f"R={refresh}")
    labels = (("Cumulative measured cost", "Seconds"),
              ("Model graph staleness", "Batches"),
              ("Cumulative snapshot exports", "MB (decimal)"))
    for ax, (title, ylabel) in zip(axes, labels):
        ax.set(title=title, xlabel="Canonical graph version", ylabel=ylabel)
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("EXP-02 | identical 10 x 10k-event replay; median of 3 measured runs", fontsize=12)
    for extension in ("svg", "png"):
        fig.savefig(asset_dir / f"exp02_cost_staleness.{extension}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/raw/exp02_normalization.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/raw/exp02_summary.json"))
    parser.add_argument("--assets", type=Path, default=Path("docs/report-assets"))
    args = parser.parse_args()
    summary = summarize(args.csv)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(args.csv, args.assets)
    print(json.dumps(summary["regime"], indent=2))


if __name__ == "__main__":
    main()
