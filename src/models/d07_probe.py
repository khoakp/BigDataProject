"""D07 hardware probe. Run from the repository root with the project venv."""

import argparse
import bisect
import csv
import gc
import io
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.graph.equivalence import checksum, full_state
from src.graph.full_builder import file_sha256, rebuild
from src.graph.normalize import normalize_state
from src.graph.snapshot import load_export
from src.models.lightgcn import LightGCN, pairwise_loss, to_sparse_tensor


ROOT = Path(__file__).resolve().parents[2]
SEED = 7
DIM = 64
LAYERS = 3
STEPS = 3
TRIPLES = 2048
LR = 0.001
MIB = 1024**2
PHASES = ("setup", "forward", "backward", "optimizer")


def toy():
    nodes = [(0, "user", 1, 0), (1, "movie", 1, 0), (2, "movie", 2, 1), (3, "user", 2, 1)]
    state = {"registry": nodes, "edges": [(0, 1, 0), (3, 2, 1), (3, 1, 2)],
             "degrees": [(0, 1), (1, 2), (2, 1), (3, 2)]}
    return {}, nodes, normalize_state(state)


def sample_triples(nodes, adjacency, limit=TRIPLES):
    """Sample deterministic pairs and exact complements in the graph's movie catalog."""
    movies = [node_id for node_id, kind, *_ in nodes if kind == "movie"]
    movie_index = {node_id: index for index, node_id in enumerate(movies)}
    occupied = defaultdict(set)
    for source, target, _ in adjacency.entries:
        if source not in movie_index:
            occupied[source].add(movie_index[target])
    rng = random.Random(SEED)
    eligible = [(user, sorted(indices)) for user, indices in occupied.items()
                if len(indices) < len(movies)]
    if not eligible:
        raise ValueError("Graph has no user with a positive and an unobserved movie")
    triples = []
    for _ in range(limit):
        user, seen = rng.choice(eligible)
        positive = movies[rng.choice(seen)]
        rank = rng.randrange(len(movies) - len(seen))
        # Find the position of the rank-th movie outside this user's positive set.
        low, high = rank, rank + len(seen)
        while low < high:
            mid = (low + high) // 2
            if mid - bisect.bisect_right(seen, mid) < rank:
                low = mid + 1
            else:
                high = mid
        triples.append((user, positive, movies[low]))
    return triples


def prepare_million(output):
    """Build the 1M smoke input from Full Rebuild without a mutable graph DB."""
    source = ROOT / "data/processed/canonical_events/canonical_1000000.parquet"
    summary = rebuild(source, output)
    if summary["events"] != 1_000_000:
        raise ValueError("Expected a one-million-event canonical prefix")
    state = full_state(output)
    adjacency = normalize_state(state)
    manifest = {
        "graph_source": "full_rebuild", "watermark_event_id": summary["events"] - 1,
        "normalization": "symmetric_degree_v1", "canonical_version": None,
        "model_graph_version": None, "registry_sha256": checksum(state["registry"]),
        "coo_sha256": checksum(adjacency.entries),
        "provenance": {"canonical_input_sha256": summary["input_sha256"]},
    }
    (output / "normalized_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, state["registry"], adjacency


def record_phase(phase, operation, device, row, step):
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    row["phase_reached"] = phase
    start = time.perf_counter()
    result = operation()
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    row["phase_timings"][phase].append({"step": step, "seconds": elapsed})
    if cuda:
        update_peaks(row, phase, device)
    return result


def update_peaks(row, phase, device):
    for name, read in (("allocated", torch.cuda.max_memory_allocated),
                       ("reserved", torch.cuda.max_memory_reserved)):
        key = f"{phase}_peak_{name}_bytes"
        row[key] = max(row.get(key, 0), read(device))


def _fit(manifest, nodes, adjacency, device, row):
    if row["events"] is not None:
        if (manifest["watermark_event_id"] + 1 != row["events"] or
                manifest["normalization"] != "symmetric_degree_v1" or
                manifest["canonical_version"] != manifest["model_graph_version"]):
            raise ValueError("Snapshot does not match the requested prefix and normalization contract")
        row.update(input_sha256=manifest["provenance"]["canonical_input_sha256"],
                   registry_sha256=manifest["registry_sha256"])
    row.update(nodes=len(nodes), nnz=len(adjacency.entries), positive_edges=len(adjacency.entries) // 2,
               graph_version=manifest.get("model_graph_version"), coo_sha256=manifest.get("coo_sha256"),
               graph_source=manifest.get("graph_source", "d06_export" if manifest else "toy"))
    triples = sample_triples(nodes, adjacency)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        row["device_name"] = torch.cuda.get_device_name(device)
        row["free_before_bytes"], row["total_vram_bytes"] = torch.cuda.mem_get_info(device)

    def setup():
        model = LightGCN(len(nodes), DIM, LAYERS).to(device)
        return (model, to_sparse_tensor(adjacency, device),
                torch.tensor(triples, dtype=torch.long, device=device),
                torch.optim.Adam(model.parameters(), lr=LR))

    model, sparse, batch, optimizer = record_phase("setup", setup, device, row, None)
    initial = model.embedding.weight.detach().clone()
    if row["dataset"] == "toy":
        # Compare against the weighted dense operator using the original D06 float64 values.
        dense = torch.zeros(adjacency.shape, dtype=torch.float64, device=device)
        for source, target, value in adjacency.entries:
            dense[source, target] = value
        current = initial.double()
        layers = [current]
        for _ in range(LAYERS):
            current = dense @ current
            layers.append(current)
        expected = torch.stack(layers).mean(dim=0)
        with torch.no_grad():
            actual = model(sparse).double()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        row["toy_dense_max_abs_diff"] = (actual - expected).abs().max().item()

    for step in range(STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = record_phase("forward", lambda: pairwise_loss(model(sparse), batch), device, row, step)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite loss")
        record_phase("backward", loss.backward, device, row, step)
        grad = model.embedding.weight.grad
        if grad is None or not bool(torch.isfinite(grad).all()) or not bool(grad.abs().max() > 0):
            raise RuntimeError("Missing, nonfinite, or zero embedding gradient")
        record_phase("optimizer", optimizer.step, device, row, step)
        row["losses"].append(float(loss.detach()))
    row["parameters_changed"] = not torch.equal(initial, model.embedding.weight.detach())
    if not row["parameters_changed"] or not bool(torch.isfinite(model.embedding.weight).all()):
        raise RuntimeError("Optimizer did not produce a finite parameter update")


def run_case(dataset, events, device_name, model_input, *, is_fallback=False):
    row = {"dataset": dataset, "events": events, "device": device_name, "is_fallback": is_fallback,
           "status": "PASS", "phase_reached": "device_check", "phase_timings": {phase: [] for phase in PHASES},
           "losses": []}
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        row.update(status="UNAVAILABLE", reason="torch.cuda.is_available() returned False")
        return row
    try:
        row["phase_reached"] = "load_snapshot"
        if callable(model_input):
            model_input = model_input()
        _fit(*model_input, device, row)
    except Exception as exc:
        row.update(status="OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "ERROR",
                   reason=f"{type(exc).__name__}: {exc}")
        if device.type == "cuda" and row["phase_reached"] in PHASES:
            try:
                update_peaks(row, row["phase_reached"], device)
            except RuntimeError as memory_error:
                row["memory_query_error"] = str(memory_error)
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except RuntimeError as cleanup_error:
            row["cleanup_error"] = str(cleanup_error)
    return row


def write_report(output, config, rows):
    rows_for_csv = [{**config, **row, "losses": json.dumps(row["losses"]),
                     "phase_timings": json.dumps(row["phase_timings"])} for row in rows]
    fields = list(dict.fromkeys(key for row in rows_for_csv for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows_for_csv)
    files = {"gpu_probe.csv": stream.getvalue(),
             "report.json": json.dumps({"config": config, "runs": rows}, indent=2)}
    for name, content in files.items():
        temporary = output / f"{name}.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            stream.write(content)
        temporary.replace(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--snapshot-100k", type=Path, help="Override the D06 v10 snapshot")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc)
    out = ROOT / "data/processed/d07_probe" / timestamp.strftime("D07-%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    d06_path = ROOT / "data/processed/d06_probe/report.json"
    config = {"run_id": out.name, "timestamp_utc": timestamp.isoformat(), "torch": str(torch.__version__),
              "torch_cuda": torch.version.cuda, "python": platform.python_version(),
              "platform": platform.platform(), "torch_cpu_threads": 1, "optimizer": "Adam",
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "seed": SEED, "embedding_dim": DIM, "layers": LAYERS, "dtype": "float32",
              "measured_steps": STEPS, "warmup_steps": 1, "triples": TRIPLES, "learning_rate": LR,
              "probe_sha256": file_sha256(Path(__file__)),
              "model_sha256": file_sha256(Path(__file__).with_name("lightgcn.py"))}
    try:
        config["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        config["nvidia_smi_error"] = str(exc)
    rows = []

    def append(row):
        rows.append(row)
        write_report(out, config, rows)
        print(f"{row['dataset']} {row['device']}: {row['status']}", flush=True)

    def load_100k():
        if args.snapshot_100k is not None:
            return load_export(args.snapshot_100k)
        report = json.loads(d06_path.read_text(encoding="utf-8"))
        if report["status"] != "PASS":
            raise ValueError("D06 report must be PASS to select its snapshot")
        return load_export(ROOT / report["artifact_directory"] / "model_graph_v10")

    def load_million():
        return prepare_million(out / "full_reference_1m")

    def skip(dataset, events, reason):
        append({"dataset": dataset, "events": events, "device": args.device, "status": "SKIPPED",
                "reason": reason, "losses": [], "phase_timings": {}})

    append(run_case("toy", None, args.device, toy))
    if rows[-1]["status"] == "PASS":
        append(run_case("ml32m_100k", 100_000, args.device, load_100k))
    else:
        skip("ml32m_100k", 100_000, "Toy probe did not pass")
    if args.device == "cuda" and rows[-1]["dataset"] == "ml32m_100k" and rows[-1]["status"] == "PASS":
        peak = max(rows[-1][f"{phase}_peak_reserved_bytes"] for phase in PHASES)
        # Leave at least 512 MiB and at least one additional 100k peak before attempting 1M.
        if rows[-1]["free_before_bytes"] - peak >= max(512*MIB, peak):
            append(run_case("ml32m_1m", 1_000_000, "cuda", load_million))
        else:
            skip("ml32m_1m", 1_000_000, "100k headroom check did not pass")
    elif args.device == "cuda":
        skip("ml32m_1m", 1_000_000, "100k CUDA probe did not pass")
    if args.device == "cuda" and not any(row["device"] == "cuda" and row["status"] == "PASS"
                                         and row["dataset"] == "ml32m_100k" for row in rows):
        append(run_case("toy", None, "cpu", toy, is_fallback=True))
        if rows[-1]["status"] == "PASS":
            append(run_case("ml32m_100k", 100_000, "cpu", load_100k, is_fallback=True))
    print(out / "gpu_probe.csv")
    return 0 if any(row["status"] == "PASS" and row["dataset"] == "ml32m_100k" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
