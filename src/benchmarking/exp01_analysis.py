"""D12: validate repeated EXP-01 raw rows and plot measured results."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(source: Path):
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty EXP-01 CSV")
    for field in ("run_id", "source_sha256", "git_commit", "config_sha256", "cache_policy"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"Mixed {field} in raw CSV")
    if any(row["status"] != "PASS" or row["correctness_pass"] != "True" for row in rows):
        raise ValueError("Raw CSV contains failed or unchecked rows")
    groups = defaultdict(dict)
    for row in rows:
        key = (int(row["history_events"]), int(row["delta_events"]), row["phase"],
               int(row["repetition"]))
        method = row["method"]
        if method in groups[key]:
            raise ValueError("Duplicate method/repetition row")
        groups[key][method] = row
    if any(set(pair) != {"full", "incremental"} for pair in groups.values()):
        raise ValueError("Unpaired Full/Incremental repetition")
    cases = defaultdict(list)
    warmups = set()
    for (history, delta, phase, repetition), pair in groups.items():
        if phase == "measured":
            if repetition < 1:
                raise ValueError("Measured repetition must start at 1")
            cases[(history, delta)].append((repetition, pair))
        elif phase != "warmup" or repetition != 0:
            raise ValueError("Unexpected phase/repetition")
        else:
            warmups.add((history, delta))
    summary = []
    if set(cases) != warmups:
        raise ValueError("Each case must have one paired warmup")
    for (history, delta), repetitions in sorted(cases.items()):
        repetitions.sort()
        indices = [rep for rep, _ in repetitions]
        if indices != list(range(1, len(repetitions) + 1)) or len(repetitions) < 2:
            raise ValueError("Measured repetitions are missing or too few")
        paired = [pair for _, pair in repetitions]
        full = [float(pair["full"]["wall_s"]) for pair in paired]
        inc = [float(pair["incremental"]["wall_s"]) for pair in paired]
        ratios = [a / b for a, b in zip(full, inc)]
        record = {"history_events": history, "delta_events": delta, "repetitions": len(repetitions),
                  "full_median_s": statistics.median(full), "full_min_s": min(full),
                  "full_max_s": max(full), "full_range_s": max(full) - min(full),
                  "incremental_median_s": statistics.median(inc), "incremental_min_s": min(inc),
                  "incremental_max_s": max(inc), "incremental_range_s": max(inc) - min(inc),
                  "paired_speedup_median": statistics.median(ratios),
                  "paired_speedup_min": min(ratios), "paired_speedup_max": max(ratios),
                  "incremental_wins": sum(b < a for a, b in zip(full, inc))}
        for method in ("full", "incremental"):
            for field in ("peak_rss_bytes", "export_bytes_written", "db_growth_bytes",
                          "app_bytes_read", "os_read_bytes", "os_write_bytes"):
                record[f"{method}_{field}_median"] = statistics.median(
                    int(pair[method][field]) for pair in paired)
        summary.append(record)
    if len({row["repetitions"] for row in summary}) != 1:
        raise ValueError("Unequal measured repetition counts across grid")
    return rows, summary


def write_summary(records, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot(records, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    deltas = sorted({row["delta_events"] for row in records})
    colors = {"full": "#195a91", "incremental": "#c45a33"}
    styles = {deltas[0]: "-", deltas[-1]: "--"}
    for metric, ylabel, filename in (("wall", "Wall time (s)", "exp01_latency.svg"),
                                    ("rss", "Sampled peak RSS (MiB)", "exp01_rss.svg"),
                                    ("bytes", "Output bytes (MiB)", "exp01_bytes.svg")):
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        for delta in deltas:
            subset = sorted((row for row in records if row["delta_events"] == delta),
                            key=lambda row: row["history_events"])
            x = [row["history_events"] / 1_000_000 for row in subset]
            for method in ("full", "incremental"):
                if metric == "wall":
                    y = [row[f"{method}_median_s"] for row in subset]
                    lo = [v - row[f"{method}_min_s"] for v, row in zip(y, subset)]
                    hi = [row[f"{method}_max_s"] - v for v, row in zip(y, subset)]
                    ax.errorbar(x, y, yerr=[lo, hi], color=colors[method],
                                linestyle=styles[delta], marker="o", capsize=3,
                                label=f"{method}, Δ={delta:,}")
                else:
                    field = "peak_rss_bytes" if metric == "rss" else "export_bytes_written"
                    y = [row[f"{method}_{field}_median"] / 2**20 for row in subset]
                    ax.plot(x, y, color=colors[method], linestyle=styles[delta], marker="o",
                            label=f"{method}, Δ={delta:,}")
        ax.set(xlabel="History H (million events)", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        fig.savefig(output_dir / filename.replace(".svg", ".png"), dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for delta in deltas:
        subset = sorted((row for row in records if row["delta_events"] == delta),
                        key=lambda row: row["history_events"])
        x = [row["history_events"] / 1_000_000 for row in subset]
        y = [row["paired_speedup_median"] for row in subset]
        lo = [v - row["paired_speedup_min"] for v, row in zip(y, subset)]
        hi = [row["paired_speedup_max"] - v for v, row in zip(y, subset)]
        ax.errorbar(x, y, yerr=[lo, hi], linestyle=styles[delta], marker="o", capsize=3,
                    label=f"Δ={delta:,}")
    ax.axhline(1, color="#555555", linewidth=1, linestyle=":")
    ax.set(xlabel="History H (million events)", ylabel="Paired Full / Incremental wall ratio")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "exp01_speedup.svg")
    fig.savefig(output_dir / "exp01_speedup.png", dpi=180)
    plt.close(fig)


def plot_crossover(core_records, scale_records, output_dir: Path):
    """Show the separate 2M run explicitly; never merge its raw rows into core."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    colors = {1000: "#195a91", 10000: "#c45a33"}
    for delta in sorted({row["delta_events"] for row in core_records}):
        core = sorted((row for row in core_records if row["delta_events"] == delta),
                      key=lambda row: row["history_events"])
        scale = sorted((row for row in scale_records if row["delta_events"] == delta),
                       key=lambda row: row["history_events"])
        x = [row["history_events"] / 1_000_000 for row in core]
        y = [row["paired_speedup_median"] for row in core]
        ax.plot(x, y, marker="o", color=colors[delta], label=f"Δ={delta:,} core run")
        for row in scale:
            sx = row["history_events"] / 1_000_000
            sy = row["paired_speedup_median"]
            ax.plot([x[-1], sx], [y[-1], sy], color=colors[delta], linestyle=":", alpha=.6)
            ax.errorbar([sx], [sy], yerr=[[sy-row["paired_speedup_min"]],
                                            [row["paired_speedup_max"]-sy]],
                        marker="s", mfc="white", mec=colors[delta], color=colors[delta],
                        capsize=3, label=f"Δ={delta:,} separate 2M run")
    ax.axhline(1, color="#555555", linewidth=1, linestyle="--")
    ax.set(xlabel="History H (million events)", ylabel="Paired Full / Incremental wall ratio")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "exp01_crossover.svg")
    fig.savefig(output_dir / "exp01_crossover.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("results/raw/exp01_full_vs_incremental.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/raw/exp01_summary_d12.csv"))
    parser.add_argument("--figures", type=Path, default=Path("docs/report-assets"))
    parser.add_argument("--scale-summary", type=Path,
                        help="Optional validated separate-run summary for the crossover figure")
    args = parser.parse_args()
    _, records = summarize(args.raw)
    write_summary(records, args.summary)
    plot(records, args.figures)
    if args.scale_summary:
        with args.scale_summary.open(newline="", encoding="utf-8") as stream:
            scale = list(csv.DictReader(stream))
        numeric = ("history_events", "delta_events", "paired_speedup_median",
                   "paired_speedup_min", "paired_speedup_max")
        scale_records = [{**row, **{key: float(row[key]) for key in numeric}} for row in scale]
        plot_crossover(records, scale_records, args.figures)
    for record in records:
        print(record["history_events"], record["delta_events"],
              round(record["full_median_s"], 4), round(record["incremental_median_s"], 4),
              round(record["paired_speedup_median"], 3), record["incremental_wins"])


if __name__ == "__main__":
    main()
