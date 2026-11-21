"""Validation-only Top-20 smoke on the historical D08 1M checkpoint.

This is an integration check for S02 utilities, not a REC-01 training or final
quality run. It never retrieves test events or test labels.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import torch

from src.evaluation.rec01 import compare_recommenders
from src.graph.equivalence import checksum, full_state
from src.graph.full_builder import file_sha256
from src.graph.normalize import normalize_state
from src.models.d08_baseline import ROOT
from src.models.lightgcn import LightGCN, to_sparse_tensor


SOURCE_SHA256 = "f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae"
TRAIN_END_TS = 840100781
VALID_END_TS = 841267862
TRAIN_EVENTS = 802988
VALID_EVENTS = 102046


def _approved_validation_events(source: Path):
    """Fetch train/validation rows only; test labels stay unread."""
    with duckdb.connect() as con:
        rows = con.execute("""SELECT event_id,event_ts,user_raw_id,movie_raw_id,rating
            FROM read_parquet(?) WHERE event_ts <= ? ORDER BY event_id""",
            [str(source), VALID_END_TS]).fetchall()
    train = [row for row in rows if row[1] <= TRAIN_END_TS]
    valid = [row for row in rows if TRAIN_END_TS < row[1] <= VALID_END_TS]
    if (len(train), len(valid)) != (TRAIN_EVENTS, VALID_EVENTS):
        raise ValueError("Approved train/validation partition counts changed")
    if [row[0] for row in rows] != list(range(TRAIN_EVENTS + VALID_EVENTS)):
        raise ValueError("Approved event IDs must be contiguous through validation")
    return train, valid


def run(source: Path, previous_run: Path, output: Path, *, device_name: str = "cpu") -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("S02 smoke output must be fresh")
    source_hash = file_sha256(source)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Canonical 1M source hash differs from approved S01 protocol")
    train, valid = _approved_validation_events(source)
    prior_report = json.loads((previous_run / "report.json").read_text(encoding="utf-8"))
    if (prior_report.get("status") != "PASS" or
            prior_report.get("source_sha256") != source_hash or
            prior_report.get("cutoffs", {}).get("train_end_ts") != TRAIN_END_TS or
            prior_report.get("cutoffs", {}).get("valid_end_ts") != VALID_END_TS or
            prior_report.get("cutoffs", {}).get("train_events") != TRAIN_EVENTS or
            prior_report.get("cutoffs", {}).get("valid_events") != VALID_EVENTS):
        raise ValueError("Historical D08 run does not match approved S01 source/cutoffs")
    train_parquet = previous_run / "train_events.parquet"
    if file_sha256(train_parquet) != prior_report["graph"]["train_input_sha256"]:
        raise ValueError("Historical train Parquet hash mismatch")
    graph_dir = previous_run / "graph"
    graph_manifest = json.loads((graph_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("registry", "edges", "degrees"):
        if file_sha256(graph_dir / f"{name}.parquet") != graph_manifest[f"{name}_sha256"]:
            raise ValueError(f"Historical graph {name} Parquet hash mismatch")
    state = full_state(previous_run / "graph")
    adjacency = normalize_state(state)
    checkpoint_path = previous_run / "model.pt"
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != prior_report["checkpoint_sha256"]:
        raise ValueError("Historical checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint["registry"] != state["registry"] or
            checkpoint["source_sha256"] != source_hash or
            checkpoint["train_end_ts"] != TRAIN_END_TS or
            checkpoint["config"] != {"dim": 64, "layers": 3, "seed": 7}):
        raise ValueError("Historical checkpoint does not match train graph/protocol")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    model = LightGCN(len(state["registry"]), 64, 3).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        embeddings = model(to_sparse_tensor(adjacency, device))
    comparison = compare_recommenders(embeddings, state, train, valid, k=20)
    report = {
        "status": "PASS", "label": "S02_VALIDATION_ONLY_SMOKE_NOT_REC01",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hash, "previous_run": str(previous_run.resolve()),
        "previous_checkpoint_sha256": checkpoint_hash,
        "registry_sha256": checksum(state["registry"]),
        "edge_sha256": checksum(state["edges"]),
        "cutoffs": {"train_end_ts": TRAIN_END_TS, "valid_end_ts": VALID_END_TS,
                    "train_events": len(train), "validation_events": len(valid)},
        "checkpoint_selection": "none; reused historical D08 final 100-step checkpoint",
        "test_labels_accessed": False, "test_metrics_computed": False,
        "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "cohort": comparison["cohort"],
        "validation_metrics_top20": comparison["metrics"],
        "ranking_users": {method: len(rows) for method, rows in comparison["rankings"].items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/processed/canonical_events/canonical_1000000.parquet")
    parser.add_argument("--previous-run", type=Path,
                        default=ROOT / "data/processed/d08_baseline_1m")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/processed/s02_probe/validation_only_d08_1m")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.previous_run, args.out,
                         device_name=args.device), indent=2))


if __name__ == "__main__":
    main()
