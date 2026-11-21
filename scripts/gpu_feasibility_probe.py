"""Measure one LightGCN workload, one GPU at a time, without training a full run.

Input is a pre-cutoff cohort directory prepared by src.evaluation.model_cohort.
No result here is a quality metric or a substitute for a T4 measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import psutil
import torch

from src.benchmarking.common import RssSampler
from src.evaluation.temporal import score_chunks
from src.graph.normalize import normalize_state
from src.models.lightgcn import LightGCN, pairwise_loss, to_sparse_tensor
from src.models.rec01_train import _sample_batches


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cohort(directory: Path) -> dict:
    """Return the existing graph API's state shape; reject malformed node mapping."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("graph_semantics") not in (None, "complete_induced_pre_cutoff_positive") or
                file_sha256(directory / "nodes.parquet") != manifest["nodes_sha256"] or
                file_sha256(directory / "edges.parquet") != manifest["edges_sha256"]):
            raise ValueError("Cohort manifest or graph semantics mismatch")
    with duckdb.connect() as con:
        nodes = con.execute("SELECT node_id,node_type,raw_id FROM read_parquet(?) ORDER BY node_id",
                            [str(directory / "nodes.parquet")]).fetchall()
        edges = con.execute("SELECT user_node_id,movie_node_id,first_event_ts FROM read_parquet(?) "
                            "ORDER BY user_node_id,movie_node_id",
                            [str(directory / "edges.parquet")]).fetchall()
    registry = [(int(node), kind, int(raw), 0) for node, kind, raw in nodes]
    edge_rows = [(int(user), int(movie), int(first_ts)) for user, movie, first_ts in edges]
    degree = Counter()
    for user, movie, _ in edge_rows:
        degree.update((user, movie))
    return {"registry": registry, "edges": edge_rows,
            "degrees": sorted(degree.items())}


def run(cohort: Path, output: Path, *, device_name: str = "cuda:0", dim: int = 64,
        layers: int = 3, batch_size: int = 1024, steps: int = 5,
        eval_users: int = 100, seed: int = 7) -> dict:
    if min(dim, batch_size, steps) <= 0 or layers < 0 or eval_users < 0:
        raise ValueError("Invalid probe configuration")
    if device_name != "cpu" and not device_name.startswith("cuda:"):
        raise ValueError("Choose one device, for example cuda:0")
    if device_name.startswith("cuda:") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; T4 result requires a CUDA run")
    cohort = Path(cohort).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    process = psutil.Process()
    report = {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": str(cohort),
        "input_sha256": {name: file_sha256(cohort / name) for name in
                         ("nodes.parquet", "edges.parquet")},
        "cohort_manifest": json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
            if (cohort / "manifest.json").is_file() else None,
        "config": {"device": device_name, "dim": dim, "layers": layers,
                   "batch_size": batch_size, "steps": steps,
                   "eval_users": eval_users, "seed": seed},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "physical_ram_bytes": psutil.virtual_memory().total,
                        "cpu_rss_start_bytes": process.memory_info().rss},
        "phases": {},
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
        report["environment"].update(gpu_name=props.name, gpu_total_bytes=total,
                                     gpu_free_start_bytes=free,
                                     gpu_count_visible=torch.cuda.device_count())

    def write() -> None:
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str),
                                            encoding="utf-8")

    def measure(name: str, operation):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        before = process.memory_info().rss
        started = time.perf_counter()
        with RssSampler(interval=.01) as sampler:
            value = operation()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        row = {"wall_s": time.perf_counter() - started,
               "cpu_rss_before_bytes": before,
               "cpu_rss_after_bytes": process.memory_info().rss,
               "cpu_peak_rss_bytes": sampler.peak}
        if device.type == "cuda":
            row.update(gpu_allocated_after_bytes=torch.cuda.memory_allocated(device),
                       gpu_reserved_after_bytes=torch.cuda.memory_reserved(device),
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       gpu_free_after_bytes=torch.cuda.mem_get_info(device)[0])
        report["phases"][name] = row
        write()
        return value

    write()
    try:
        state = measure("load_python_graph", lambda: load_cohort(cohort))
        report["graph"] = {"nodes": len(state["registry"]),
                           "positive_edges": len(state["edges"]),
                           "sparse_entries": 2 * len(state["edges"])}
        adjacency = measure("normalize_python", lambda: normalize_state(state))
        generator = _sample_batches(state, batch_size, steps, seed)
        first_batch = measure("prepare_sampler", lambda: next(generator))
        sparse = measure("to_sparse_coo", lambda: to_sparse_tensor(adjacency, device))
        model = LightGCN(len(state["registry"]), dim, layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        report["graph"]["sparse_tensor_nnz"] = sparse._nnz()
        embeddings = measure("forward", lambda: model(sparse))
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("Nonfinite forward embeddings")
        del embeddings

        def step(batch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            triples = torch.tensor(batch, dtype=torch.long, device=device)
            loss = pairwise_loss(model(sparse), triples)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite BPR loss")
            loss.backward()
            optimizer.step()
            return float(loss.detach().cpu())

        report["first_step_loss"] = measure("first_backward_adam_step",
                                           lambda: step(first_batch))
        if steps > 1:
            report["remaining_step_losses"] = measure(
                "short_trajectory", lambda: [step(next(generator)) for _ in range(steps - 1)])

        checkpoint = output / "probe_checkpoint.pt"
        def save_checkpoint():
            torch.save({"state_dict": model.state_dict(), "registry": state["registry"],
                        "config": report["config"]}, checkpoint)
            return checkpoint.stat().st_size
        report["checkpoint_bytes"] = measure("checkpoint", save_checkpoint)

        if eval_users:
            seen = {}
            for user, movie, _ in state["edges"]:
                seen.setdefault(user, set()).add(movie)
            users = {raw: node for node, kind, raw, _ in state["registry"] if kind == "user"}
            movies = {raw: node for node, kind, raw, _ in state["registry"] if kind == "movie"}
            inverse_users = {node: raw for raw, node in users.items()}
            inverse_movies = {node: raw for raw, node in movies.items()}
            selected_nodes = sorted(seen)[:eval_users]
            selected_users = {inverse_users[node]: node for node in selected_nodes}
            selected_seen = {inverse_users[node]: {inverse_movies[movie_node]
                                                   for movie_node in seen[node]}
                             for node in selected_nodes}
            model.eval()
            with torch.inference_mode():
                final_embeddings = measure("evaluation_forward", lambda: model(sparse))
                rankings = measure("rank_train_catalog", lambda: score_chunks(
                    final_embeddings, final_embeddings, selected_users, movies,
                    selected_seen, k=20, chunk_size=1024))
            report["ranking"] = {"users": len(rankings), "candidate_movies": len(movies),
                                 "returned_items": sum(map(len, rankings.values()))}

        if device.type == "cuda":
            total = report["environment"]["gpu_total_bytes"]
            peak = max(row.get("gpu_peak_reserved_bytes", 0)
                       for row in report["phases"].values())
            report["gpu_peak_reserved_bytes"] = peak
            report["gpu_total_headroom_fraction"] = (total - peak) / total
            report["gpu_free_end_bytes"] = torch.cuda.mem_get_info(device)[0]
            report["gpu_effective_headroom_fraction"] = min(
                report["gpu_total_headroom_fraction"],
                report["gpu_free_end_bytes"] / total)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--eval-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(args.cohort, args.out, device_name=args.device, dim=args.dim,
                 layers=args.layers, batch_size=args.batch_size, steps=args.steps,
                 eval_users=args.eval_users, seed=args.seed)
    print(json.dumps({"status": result["status"], "graph": result["graph"],
                      "gpu_peak_reserved_bytes": result.get("gpu_peak_reserved_bytes"),
                      "report": str(args.out / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
