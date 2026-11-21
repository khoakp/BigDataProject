"""D08 strict temporal LightGCN baseline on a canonical prefix."""
import argparse
import json
import platform
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import torch

from src.evaluation.temporal import cutoff_cohort, evaluate_topk, score_chunks, split_events
from src.graph.equivalence import compare_states, full_state, incremental_state
from src.graph.full_builder import file_sha256, rebuild
from src.graph.incremental import IncrementalGraph
from src.graph.normalize import normalize_state
from src.models.lightgcn import LightGCN, pairwise_loss, to_sparse_tensor


ROOT = Path(__file__).resolve().parents[2]


def load_events(path):
    with duckdb.connect() as con:
        return con.execute("SELECT event_id,event_ts,user_raw_id,movie_raw_id,rating FROM read_parquet(?) ORDER BY event_id", [str(path)]).fetchall()


def choose_cutoffs(events):
    timestamps = sorted({event[1] for event in events})
    if len(timestamps) < 3:
        raise ValueError("At least three distinct timestamps are required")
    train_index = min(max(0, int(len(timestamps) * .8) - 1), len(timestamps) - 3)
    valid_index = min(max(train_index + 1, int(len(timestamps) * .9) - 1), len(timestamps) - 2)
    return timestamps[train_index], timestamps[valid_index]


def train_only_graph(source, output, train_count):
    target = output / "train_events.parquet"
    source_sql = str(source.resolve()).replace("\\", "/").replace("'", "''")
    target_sql = str(target.resolve()).replace("\\", "/").replace("'", "''")
    with duckdb.connect() as con:
        con.execute(f"COPY (SELECT * FROM read_parquet('{source_sql}') WHERE event_id < {int(train_count)} "
                    f"ORDER BY event_id) TO '{target_sql}' (FORMAT PARQUET)")
    summary = rebuild(target, output / "graph")
    state = full_state(output / "graph")
    if any(first >= train_count for *_, first in state["edges"]):
        raise AssertionError("Future edge in train graph")
    return summary, state, normalize_state(state)


def incremental_train_graph(source, output, train_count, batch_events=10_000):
    """Apply only train events to versioned D04 state, then compare full reference."""
    summary, reference, _ = train_only_graph(source, output, train_count)
    train_path = output / "train_events.parquet"
    source_sql = str(train_path.resolve()).replace("\\", "/").replace("'", "''")
    with IncrementalGraph(output / "graph.duckdb") as graph:
        for batch_id, start in enumerate(range(0, train_count, batch_events)):
            target = output / f"batch_{batch_id:05d}.parquet"
            target_sql = str(target.resolve()).replace("\\", "/").replace("'", "''")
            with duckdb.connect() as con:
                con.execute(f"COPY (SELECT * FROM read_parquet('{source_sql}') WHERE event_id >= {start} "
                            f"AND event_id < {min(start + batch_events, train_count)} ORDER BY event_id) "
                            f"TO '{target_sql}' (FORMAT PARQUET)")
            graph.apply_batch(batch_id, target)
        state = incremental_state(graph)
        ledger = graph.con.execute("SELECT max(graph_version),max(max_event_id) FROM applied_batches").fetchone()
    comparison = compare_states(reference, state)
    if not comparison["pass"] or ledger[1] != train_count - 1:
        raise AssertionError("Incremental train state differs from full reference")
    return summary, state, normalize_state(state), ledger[0], comparison


def sample_batches(state, batch_size, steps, seed):
    users = defaultdict(set)
    movies = [node for node, kind, *_ in state["registry"] if kind == "movie"]
    for user, movie, _ in state["edges"]:
        users[user].add(movie)
    eligible = [user for user in users if len(users[user]) < len(movies)]
    if not eligible:
        raise ValueError("No train user has an unobserved train movie")
    rng = random.Random(seed)
    for _ in range(steps):
        batch = []
        while len(batch) < batch_size:
            user = rng.choice(eligible)
            positive = rng.choice(sorted(users[user]))
            negative = rng.choice(movies)
            if negative not in users[user]:
                batch.append((user, positive, negative))
        yield batch


def run(source, output, *, steps=100, batch_size=1024, dim=64, layers=3,
        lr=.001, seed=7, k=10, chunk_size=1024, device_name="cpu", graph_method="full"):
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    output.mkdir(parents=True, exist_ok=True)
    events = load_events(source)
    train_end, valid_end = choose_cutoffs(events)
    train, valid, test = split_events(events, train_end, valid_end)
    if graph_method == "incremental":
        summary, state, adjacency, graph_version, comparison = incremental_train_graph(source, output, len(train))
    elif graph_method == "full":
        summary, state, adjacency = train_only_graph(source, output, len(train))
        graph_version, comparison = None, None
    else:
        raise ValueError("Unknown graph method")
    registry = state["registry"]
    valid_truth, seen, valid_cohort = cutoff_cohort(train, valid, registry)
    test_truth, _, test_cohort = cutoff_cohort(train, test, registry)
    node_users = {raw: node for node, kind, raw, *_ in registry if kind == "user"}
    node_movies = {raw: node for node, kind, raw, *_ in registry if kind == "movie"}
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = LightGCN(len(registry), dim, layers).to(device)
    sparse = to_sparse_tensor(adjacency, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for batch in sample_batches(state, batch_size, steps, seed):
        optimizer.zero_grad(set_to_none=True)
        triples = torch.tensor(batch, dtype=torch.long, device=device)
        loss = pairwise_loss(model(sparse), triples)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite BPR loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        embeddings = model(sparse)
    topk_valid = score_chunks(embeddings, embeddings, {u: node_users[u] for u in valid_truth}, node_movies, seen, k, chunk_size)
    topk_test = score_chunks(embeddings, embeddings, {u: node_users[u] for u in test_truth}, node_movies, seen, k, chunk_size)
    metrics = {"valid": evaluate_topk(topk_valid, valid_truth, k),
               "test": evaluate_topk(topk_test, test_truth, k)}
    checkpoint = output / "model.pt"
    torch.save({"state_dict": model.state_dict(), "registry": registry,
                "train_end_ts": train_end, "source_sha256": file_sha256(source),
                "config": {"dim": dim, "layers": layers, "seed": seed},
                "graph_version": graph_version}, checkpoint)
    recommendations = [{"split": split, "user_raw_id": user, "movie_raw_id": movie,
                        "rank": rank, "score": score}
                       for split, topk, truth in (("valid", topk_valid, valid_truth), ("test", topk_test, test_truth))
                       for user in sorted(truth) for rank, (movie, score) in enumerate(topk[user], 1)]
    (output / "topk_raw.json").write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
    report = {"status": "PASS", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": file_sha256(source), "python": platform.python_version(),
              "torch": torch.__version__, "device": str(device), "seed": seed,
              "config": {"steps": steps, "batch_size": batch_size, "dim": dim, "layers": layers,
                         "lr": lr, "k": k, "chunk_size": chunk_size},
              "cutoffs": {"train_end_ts": train_end, "valid_end_ts": valid_end,
                          "train_events": len(train), "valid_events": len(valid), "test_events": len(test)},
              "graph": {"nodes": summary["nodes"], "edges": summary["edges"],
                        "train_input_sha256": summary["input_sha256"],
                        "method": graph_method, "version": graph_version,
                        "full_equivalence": comparison},
              "candidate_policy": "train positive catalog; mask train positive seen; no future negatives",
              "cohorts": {"valid": valid_cohort, "test": test_cohort},
              "metrics": metrics, "loss_first": losses[0], "loss_last": losses[-1],
              "checkpoint_sha256": file_sha256(checkpoint)}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/canonical_events/canonical_100000.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/d08_baseline")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out, steps=args.steps, device_name=args.device)["metrics"], indent=2))
