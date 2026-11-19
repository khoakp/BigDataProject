"""Reproducible D04 integration probe, with isolated temporary database per run."""
import hashlib
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .incremental import IncrementalGraph


def fingerprint(state):
    return hashlib.sha256(json.dumps(state, separators=(",", ":")).encode()).hexdigest()


def run():
    source = Path("data/processed/canonical_events/canonical_100000.parquet")
    report = {"run_id": "D04-100k-10k", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "duckdb": duckdb.__version__,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "code_state": "uncommitted working tree", "seed": "not applicable",
              "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "threshold": 4.0, "batch_size": 10000, "batches": []}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        database = base / "state.duckdb"
        with duckdb.connect() as con, IncrementalGraph(database) as graph:
            old_registry = []
            for batch_id in range(10):
                batch = base / f"batch_{batch_id}.parquet"
                con.execute(f"COPY (SELECT * FROM read_parquet('{source.as_posix()}') WHERE event_id >= {batch_id * 10000} AND event_id < {(batch_id+1)*10000} ORDER BY event_id) TO '{batch.as_posix()}' (FORMAT PARQUET)")
                row = graph.apply_batch(batch_id, batch)
                state = graph.snapshot()
                assert state["node_registry"][:len(old_registry)] == old_registry
                old_registry = state["node_registry"]
                assert graph.apply_batch(batch_id, batch)["status"] == "NO_OP"
                assert graph.snapshot() == state
                report["batches"].append(row)
            edges = len(state["edge_state"])
            types = {r[0]: r[1] for r in state["node_registry"]}
            user_sum = sum(r[1] for r in state["degree_state"] if types[r[0]] == "user")
            movie_sum = sum(r[1] for r in state["degree_state"] if types[r[0]] == "movie")
            assert user_sum == edges == movie_sum
            expected = {}
            for user, movie, *_ in state["edge_state"]:
                expected[user] = expected.get(user, 0) + 1
                expected[movie] = expected.get(movie, 0) + 1
            assert expected == {node: degree for node, degree, _ in state["degree_state"]}
            report.update(edges=edges, nodes=len(types), sum_user_degree=user_sum,
                          sum_movie_degree=movie_sum, state_sha256=fingerprint(state),
                          replay_noops=10, stable_ids=True, degree_recount_pass=True)
        with IncrementalGraph(database) as graph:
            assert graph.snapshot() == state
            assert graph.apply_batch(9, batch)["status"] == "NO_OP"
            assert graph.snapshot() == state
        report.update(reopen_pass=True, status="PASS")
    out = Path("data/processed/d04_probe/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
