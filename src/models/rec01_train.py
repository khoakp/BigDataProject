"""Train REC-01 LightGCN, select on validation, then compare on held-out test.

The default protocol is the Project Owner-approved MovieLens 32M canonical 1M
prefix. A custom Protocol is accepted by ``run`` solely for small integration
tests; the CLI always uses the approved protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import torch

from src.evaluation.checkpoint_selection import select_checkpoint
from src.evaluation.rec01 import compare_recommenders
from src.graph.full_builder import file_sha256
from src.graph.incremental import IncrementalGraph
from src.graph.normalize import compare_adjacency
from src.graph.snapshot import export_snapshot, load_export, materialize_normalized
from src.models.d08_baseline import ROOT, incremental_train_graph
from src.models.lightgcn import LightGCN, pairwise_loss, to_sparse_tensor


@dataclass(frozen=True)
class Protocol:
    source_sha256: str
    events: int
    distinct_timestamps: int
    train_end_ts: int
    valid_end_ts: int
    train_events: int
    valid_events: int
    test_events: int
    checkpoint_steps: tuple[int, ...] = (100, 300, 500)
    batch_size: int = 1024
    dim: int = 64
    layers: int = 3
    lr: float = 0.001
    seed: int = 7
    k: int = 20
    chunk_size: int = 1024
    graph_batch_events: int = 10_000


APPROVED = Protocol(
    source_sha256="f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae",
    events=1_000_000, distinct_timestamps=478_972,
    train_end_ts=840100781, valid_end_ts=841267862,
    train_events=802_988, valid_events=102_046, test_events=94_966,
)


CSV_FIELDS = ("run_id", "split", "method", "checkpoint_step", "users", "warm_pairs",
              "candidate_movies", "recall_20", "ndcg_20", "source_sha256",
              "config_sha256", "graph_version", "checkpoint_sha256", "device", "status")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _code_provenance() -> dict:
    paths = (Path(__file__), ROOT / "src/models/lightgcn.py",
             ROOT / "src/models/d08_baseline.py", ROOT / "src/evaluation/temporal.py",
             ROOT / "src/evaluation/rec01.py", ROOT / "src/evaluation/baselines.py",
             ROOT / "src/evaluation/checkpoint_selection.py",
             ROOT / "src/graph/incremental.py", ROOT / "src/graph/normalize.py",
             ROOT / "src/graph/snapshot.py", ROOT / "src/graph/full_builder.py",
             ROOT / "src/graph/equivalence.py")
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in paths}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {"git_head": head, "code_sha256": hashes}


def _check_protocol(source: Path, protocol: Protocol) -> None:
    """Validate metadata without fetching held-out rating/user/movie labels."""
    actual_hash = file_sha256(source)
    if actual_hash != protocol.source_sha256:
        raise ValueError("REC-01 source SHA-256 differs from the approved protocol")
    with duckdb.connect() as con:
        total, distinct_ts, distinct_ids, min_id, max_id = con.execute(
            "SELECT count(*),count(DISTINCT event_ts),count(DISTINCT event_id),"
            "min(event_id),max(event_id) FROM read_parquet(?)", [str(source)]).fetchone()
        if (total, distinct_ids, min_id, max_id) != (protocol.events, protocol.events,
                                                     0, protocol.events - 1):
            raise ValueError("REC-01 event IDs/count differ from the approved protocol")
        if distinct_ts != protocol.distinct_timestamps:
            raise ValueError("REC-01 timestamp count differs from the approved protocol")
        if distinct_ts < 3:
            raise ValueError("REC-01 needs at least three distinct timestamps")
        train_index = min(max(0, int(distinct_ts * .8) - 1), distinct_ts - 3)
        valid_index = min(max(train_index + 1, int(distinct_ts * .9) - 1), distinct_ts - 2)
        actual_cutoffs = tuple(con.execute(
            "SELECT event_ts FROM (SELECT DISTINCT event_ts FROM read_parquet(?)) "
            "ORDER BY event_ts LIMIT 1 OFFSET ?", [str(source), index]).fetchone()[0]
            for index in (train_index, valid_index))
        counts = con.execute(
            "SELECT count(*) FILTER (WHERE event_ts<=?),"
            "count(*) FILTER (WHERE event_ts>? AND event_ts<=?),"
            "count(*) FILTER (WHERE event_ts>?) FROM read_parquet(?)",
            [protocol.train_end_ts, protocol.train_end_ts,
             protocol.valid_end_ts, protocol.valid_end_ts, str(source)]).fetchone()
    if actual_cutoffs != (protocol.train_end_ts, protocol.valid_end_ts):
        raise ValueError("REC-01 temporal cutoffs differ from the approved protocol")
    if counts != (protocol.train_events, protocol.valid_events, protocol.test_events):
        raise ValueError("REC-01 temporal partition counts differ from the approved protocol")


def _load_partition(source: Path, valid_end_ts: int, *, test: bool) -> list:
    """Fetch rating labels only for the requested side of the validation cutoff."""
    condition = "event_ts>?" if test else "event_ts<=?"
    with duckdb.connect() as con:
        return con.execute(
            "SELECT event_id,event_ts,user_raw_id,movie_raw_id,rating "
            f"FROM read_parquet(?) WHERE {condition} ORDER BY event_id",
            [str(source), valid_end_ts]).fetchall()


def _train_validation(source: Path, protocol: Protocol) -> tuple[list, list]:
    events = _load_partition(source, protocol.valid_end_ts, test=False)
    if len(events) != protocol.train_events + protocol.valid_events:
        raise ValueError("REC-01 event/timestamp count differs from the approved protocol")
    if any(event_id != index for index, (event_id, *_) in enumerate(events)):
        raise ValueError("REC-01 train/validation event IDs are not contiguous")
    train = [event for event in events if event[1] <= protocol.train_end_ts]
    valid = [event for event in events if protocol.train_end_ts < event[1] <= protocol.valid_end_ts]
    if (len(train), len(valid)) != (protocol.train_events, protocol.valid_events):
        raise ValueError("REC-01 train/validation row counts changed")
    return train, valid


def _rank_rows(run_id: str, split: str, method: str, rankings: dict) -> list[dict]:
    return [{"run_id": run_id, "split": split, "method": method,
             "user_raw_id": user, "rank": rank, "movie_raw_id": movie, "score": score}
            for user in sorted(rankings)
            for rank, (movie, score) in enumerate(rankings[user], 1)]


def _csv_row(run_id: str, split: str, method: str, step: int | None,
             comparison: dict, source_hash: str, config_hash: str, graph_version: int,
             checkpoint_hash: str | None, device_name: str) -> dict:
    cohort, metric = comparison["cohort"], comparison["metrics"][method]
    return {"run_id": run_id, "split": split, "method": method,
            "checkpoint_step": step if step is not None else "",
            "users": metric["users"], "warm_pairs": cohort["warm_pairs"],
            "candidate_movies": cohort["catalog_movies"],
            "recall_20": metric["recall"], "ndcg_20": metric["ndcg"],
            "source_sha256": source_hash, "config_sha256": config_hash,
            "graph_version": graph_version,
            "checkpoint_sha256": checkpoint_hash or "", "device": device_name,
            "status": "PASS" if metric["users"] else "UNAVAILABLE"}


def _save_checkpoint(path: Path, model: LightGCN, *, protocol: Protocol,
                     source_hash: str, config_hash: str, graph_version: int,
                     registry: list, step: int) -> str:
    torch.save({"state_dict": model.state_dict(), "registry": registry,
                "source_sha256": source_hash, "config_sha256": config_hash,
                "graph_version": graph_version, "checkpoint_step": step,
                "checkpoint_train_cutoff": protocol.train_end_ts,
                "model_graph_version": graph_version,
                "config": {"dim": protocol.dim, "layers": protocol.layers,
                           "seed": protocol.seed, "positive_threshold": 4.0}}, path)
    return file_sha256(path)


def _sample_batches(state: dict, batch_size: int, steps: int, seed: int):
    """D08 sampling sequence with sorted positives cached once per user."""
    positives = defaultdict(set)
    movies = [node for node, kind, *_ in state["registry"] if kind == "movie"]
    for user, movie, _ in state["edges"]:
        positives[user].add(movie)
    eligible = [user for user in positives if len(positives[user]) < len(movies)]
    if not eligible:
        raise ValueError("No train user has an unobserved train movie")
    ordered = {user: sorted(items) for user, items in positives.items()}
    rng = random.Random(seed)
    for _ in range(steps):
        batch = []
        while len(batch) < batch_size:
            user = rng.choice(eligible)
            positive = rng.choice(ordered[user])
            negative = rng.choice(movies)
            if negative not in positives[user]:
                batch.append((user, positive, negative))
        yield batch


def run(source: Path, output: Path, *, device_name: str = "cuda",
        protocol: Protocol = APPROVED) -> dict:
    """Run one fixed trajectory; test is evaluated only after checkpoint reload."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("REC-01 output directory must be fresh")
    if (not protocol.checkpoint_steps or
            tuple(sorted(set(protocol.checkpoint_steps))) != protocol.checkpoint_steps or
            min(protocol.checkpoint_steps) <= 0 or protocol.k != 20):
        raise ValueError("Invalid REC-01 checkpoint schedule or K")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; specify --device cpu for an explicit fallback")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    config = asdict(protocol)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report = {"run_id": output.name, "status": "RUNNING",
              "started_utc": started.isoformat(), "protocol": config,
              "config_sha256": config_hash,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "torch_cuda": torch.version.cuda, "duckdb": duckdb.__version__,
                              "platform": platform.platform(), "device": device_name,
                              "gpu": torch.cuda.get_device_name(0) if device_name == "cuda" else None,
                              "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")},
              **_code_provenance()}
    _write_json(output / "report.json", report)
    try:
        _check_protocol(source, protocol)
        train, valid = _train_validation(source, protocol)
        source_hash = protocol.source_sha256
        report["source"] = {"path": str(source.resolve()), "sha256": source_hash,
                            "events": protocol.events, "cutoffs": [protocol.train_end_ts,
                                                                   protocol.valid_end_ts],
                            "partition_events": [protocol.train_events,
                                                 protocol.valid_events, protocol.test_events]}

        graph_start = time.perf_counter()
        (output / "graph_build").mkdir()
        summary, state, adjacency, graph_version, comparison = incremental_train_graph(
            source, output / "graph_build", len(train), protocol.graph_batch_events)
        if not comparison["pass"]:
            raise AssertionError("Train graph does not match Full Rebuild")
        with IncrementalGraph(output / "graph_build" / "graph.duckdb") as graph:
            model_graph_manifest = materialize_normalized(
                graph, expected_version=graph_version,
                provenance={"dataset_id": f"ml32m_canonical_{protocol.events}",
                            "canonical_input_sha256": source_hash,
                            "config_sha256": config_hash})
            export_snapshot(graph, graph_version, output / "model_graph")
        _, exported_nodes, exported_adjacency = load_export(output / "model_graph")
        if (not compare_adjacency(adjacency, exported_adjacency)["pass"] or
                len(exported_nodes) != len(state["registry"])):
            raise AssertionError("Exported model graph differs from train adjacency")
        report["graph"] = {"version": graph_version, "nodes": summary["nodes"],
                           "edges": summary["edges"], "full_equivalence": comparison,
                           "build_s": time.perf_counter() - graph_start,
                           "normalized_nnz": len(adjacency.entries),
                           "model_graph_coo_sha256": model_graph_manifest["coo_sha256"],
                           "model_graph_manifest_sha256": file_sha256(
                               output / "model_graph" / "manifest.json")}
        registry = state["registry"]
        _write_json(output / "model_ids.json", [
            {"model_index": node_id, "node_type": kind, "raw_id": raw_id,
             "first_event_id": first_event_id}
            for node_id, kind, raw_id, first_event_id in registry])

        device = torch.device(device_name)
        torch.manual_seed(protocol.seed)
        torch.set_num_threads(1)
        model = LightGCN(len(registry), protocol.dim, protocol.layers).to(device)
        sparse = to_sparse_tensor(adjacency, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol.lr)
        records = []
        comparisons = {}
        checkpoint_hashes = {}
        losses = {}
        train_start = time.perf_counter()
        for step, batch in enumerate(_sample_batches(state, protocol.batch_size,
                                                     protocol.checkpoint_steps[-1],
                                                     protocol.seed), 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            triples = torch.tensor(batch, dtype=torch.long, device=device)
            loss = pairwise_loss(model(sparse), triples)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite BPR loss at step {step}")
            loss.backward()
            optimizer.step()
            if step not in protocol.checkpoint_steps:
                continue
            losses[step] = float(loss.detach())
            checkpoint = output / f"model_step_{step:04d}.pt"
            checkpoint_hashes[step] = _save_checkpoint(
                checkpoint, model, protocol=protocol, source_hash=source_hash,
                config_hash=config_hash, graph_version=graph_version,
                registry=registry, step=step)
            model.eval()
            with torch.no_grad():
                embeddings = model(sparse)
            validation = compare_recommenders(embeddings, state, train, valid,
                                              k=protocol.k, chunk_size=protocol.chunk_size)
            metrics = validation["metrics"]["lightgcn"]
            if metrics["users"] == 0:
                raise ValueError("Validation warm cohort is empty; cannot select checkpoint")
            records.append({"step": step, "validation_recall_20": metrics["recall"],
                            "validation_ndcg_20": metrics["ndcg"],
                            "checkpoint_sha256": checkpoint_hashes[step]})
            comparisons[step] = validation
            report["validation_candidates"] = records
            report["checkpoint_sha256"] = checkpoint_hashes
            _write_json(output / "report.json", report)

        report["trajectory_s"] = time.perf_counter() - train_start
        report["trajectory_scope"] = "optimizer steps + checkpoint save + validation scoring"
        selected = select_checkpoint(records)
        selected_step = selected["step"]
        selected_path = output / f"model_step_{selected_step:04d}.pt"
        if file_sha256(selected_path) != selected["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        saved = torch.load(selected_path, map_location=device, weights_only=False)
        if (saved["registry"] != registry or saved["source_sha256"] != source_hash or
                saved["config_sha256"] != config_hash or
                saved["checkpoint_step"] != selected_step or
                saved["graph_version"] != graph_version or
                saved["checkpoint_train_cutoff"] != protocol.train_end_ts):
            raise ValueError("Selected checkpoint metadata mismatch")
        reloaded = LightGCN(len(registry), protocol.dim, protocol.layers).to(device)
        reloaded.load_state_dict(saved["state_dict"])
        reloaded.eval()
        with torch.no_grad():
            embeddings = reloaded(sparse)
        valid_reload = compare_recommenders(embeddings, state, train, valid,
                                            k=protocol.k, chunk_size=protocol.chunk_size)
        original_ranking = comparisons[selected_step]["rankings"]["lightgcn"]
        if {user: [movie for movie, _ in rows] for user, rows in valid_reload["rankings"]["lightgcn"].items()} != {
                user: [movie for movie, _ in rows] for user, rows in original_ranking.items()}:
            raise AssertionError("Reloaded checkpoint changes validation Top-20 movie IDs")

        # Fetch held-out labels only after checkpoint selection and reload checks.
        heldout_test = _load_partition(source, protocol.valid_end_ts, test=True)
        if (len(heldout_test) != protocol.test_events or
                any(event_id != protocol.train_events + protocol.valid_events + index
                    for index, (event_id, *_) in enumerate(heldout_test))):
            raise ValueError("REC-01 test event IDs/count changed after selection")
        test = compare_recommenders(embeddings, state, train, heldout_test,
                                    k=protocol.k, chunk_size=protocol.chunk_size)
        if test["metrics"]["lightgcn"]["users"] == 0:
            raise ValueError("Test warm cohort is empty; REC-01 quality is unavailable")
        selected_validation = comparisons[selected_step]
        csv_rows = [_csv_row(output.name, "validation", "lightgcn", row["step"],
                             comparisons[row["step"]], source_hash, config_hash, graph_version,
                             checkpoint_hashes[row["step"]], device_name) for row in records]
        csv_rows += [_csv_row(output.name, "validation", "popularity", None,
                              selected_validation, source_hash, config_hash, graph_version,
                              None, device_name),
                     _csv_row(output.name, "test", "lightgcn", selected_step,
                              test, source_hash, config_hash, graph_version,
                              checkpoint_hashes[selected_step], device_name),
                     _csv_row(output.name, "test", "popularity", None,
                              test, source_hash, config_hash, graph_version,
                              None, device_name)]
        with (output / "rec01_quality.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        rankings = []
        for split, result in (("validation", selected_validation), ("test", test)):
            for method, ranking in result["rankings"].items():
                rankings.extend(_rank_rows(output.name, split, method, ranking))
        _write_json(output / "rankings.json", rankings)
        report.update(status="PASS", selected_checkpoint=selected,
                      selected_checkpoint_file=selected_path.name,
                      selected_checkpoint_reloaded=True,
                      losses_at_checkpoints=losses,
                      cohorts={"validation": selected_validation["cohort"],
                               "test": test["cohort"]},
                      metrics={"validation": selected_validation["metrics"],
                               "test": test["metrics"]},
                      raw_csv="rec01_quality.csv", rankings="rankings.json",
                      model_ids_sha256=file_sha256(output / "model_ids.json"))
    except Exception as exc:
        report["status"] = "ERROR"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/processed/canonical_events/canonical_1000000.parquet")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/processed/rec01" /
                        datetime.now(timezone.utc).strftime("REC01-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out, device_name=args.device), indent=2, default=str))


if __name__ == "__main__":
    main()
