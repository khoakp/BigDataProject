"""D10 clean MovieLens CSV -> canonical events -> versioned LightGCN MVP."""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import duckdb

from src.graph.full_builder import file_sha256
from src.ingest.d02_profile import run as canonicalize
from src.models.d09_pipeline import run as pipeline
from src.models.d08_baseline import ROOT


ARTIFACTS = ("canonical/manifest.json",
             "pipeline/graph.duckdb", "pipeline/graph/registry.parquet",
             "pipeline/graph/edges.parquet", "pipeline/graph/degrees.parquet",
             "pipeline/model.pt", "pipeline/model_ids.json", "pipeline/report.json",
             "pipeline/integration_report.json", "pipeline/topk_raw.json",
             "pipeline/topk_enriched.json")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run(ratings: Path, movies: Path, output: Path, *, device_name="cuda", steps=100,
        prefix_events=100_000, fallback_cpu=True):
    """Use a new output directory; never present a partial run as a successful MVP."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Clean run directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    manifest = {"status": "RUNNING", "run_id": output.name, "started_utc": started.isoformat(),
                "source": {"ratings": str(ratings.resolve()), "movies": str(movies.resolve())},
                "config": {"prefix_events": prefix_events, "steps": steps,
                           "requested_device": device_name, "fallback_cpu": fallback_cpu},
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "torch_cuda": torch.version.cuda, "duckdb": duckdb.__version__,
                                "platform": platform.platform(),
                                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "attempts": []}
    manifest["config_sha256"] = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()
    manifest["code_sha256"] = {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
                               for path in (ROOT / "src/ingest/d02_profile.py",
                                            ROOT / "src/graph/incremental.py",
                                            ROOT / "src/graph/normalize.py",
                                            ROOT / "src/models/lightgcn.py",
                                            ROOT / "src/models/d08_baseline.py",
                                            ROOT / "src/models/d09_pipeline.py",
                                            Path(__file__))}
    target = output / "run_manifest.json"
    write_json(target, manifest)
    try:
        canonical_dir = output / "canonical"
        profile = canonicalize(ratings, canonical_dir, max_prefix=prefix_events)
        manifest["canonical"] = {"events": prefix_events,
                                 "source_sha256": profile["source_sha256"],
                                 "prefix_sha256": profile["prefixes"][str(prefix_events)]["sha256"]}
        manifest["source"]["ratings_sha256"] = profile["source_sha256"]
        source = canonical_dir / f"canonical_{prefix_events}.parquet"
        devices = [device_name]
        if device_name == "cuda" and fallback_cpu:
            devices.append("cpu")
        integration = None
        selected = None
        for attempt_number, device in enumerate(devices):
            # A failed CUDA attempt gets its own directory; CPU never resumes its graph DB.
            attempt_dir = output / ("pipeline" if attempt_number == 0 else "pipeline_cpu_fallback")
            try:
                integration = pipeline(source, movies, attempt_dir, device_name=device, steps=steps)
                manifest["attempts"].append({"device": device, "status": "PASS"})
                selected = attempt_dir
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                status = "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() else "ERROR"
                manifest["attempts"].append({"device": device, "status": status,
                                             "reason": f"{type(exc).__name__}: {exc}"})
                write_json(target, manifest)
                if device != "cuda" or not fallback_cpu:
                    raise
        if integration is None or selected is None:
            raise RuntimeError("No successful pipeline attempt")
        checkpoint = torch.load(selected / "model.pt", map_location="cpu", weights_only=False)
        node_map = [{"model_index": node_id, "node_type": kind, "raw_id": raw_id,
                     "first_event_id": first_event_id}
                    for node_id, kind, raw_id, first_event_id in checkpoint["registry"]]
        if [row["model_index"] for row in node_map] != list(range(len(node_map))):
            raise AssertionError("Model indices are not contiguous")
        if checkpoint["graph_version"] != integration["model_checkpoint_version"]:
            raise AssertionError("Checkpoint graph version differs from integration report")
        write_json(selected / "model_ids.json", node_map)
        manifest["source"]["movies_sha256"] = integration["movies_source_sha256"]
        if selected.name != "pipeline":
            # Report actual selected location, do not rename a failed CUDA directory.
            manifest["selected_pipeline"] = selected.name
        else:
            manifest["selected_pipeline"] = "pipeline"
        required = [f"canonical/canonical_{prefix_events}.parquet"] + [
            item.replace("pipeline/", f"{selected.name}/") for item in ARTIFACTS]
        manifest["artifacts"] = {item: {"sha256": file_sha256(output / item),
                                         "bytes": (output / item).stat().st_size}
                                 for item in required}
        manifest["model"] = {"node_count": len(node_map),
                             "graph_version": checkpoint["graph_version"],
                             "checkpoint_sha256": integration["checkpoint_sha256"]}
        manifest["integration"] = {"metadata_rows": integration["topk_coverage"]["rows"],
                                   "metrics": integration["metrics"],
                                   "ranking_fields_unchanged": integration["ranking_fields_unchanged"],
                                   "metrics_unchanged": integration["metrics_unchanged"]}
        manifest["status"] = "PASS"
    except Exception as exc:
        manifest["status"] = "ERROR"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(target, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=ROOT.parent / "ml-32m/ratings.csv")
    parser.add_argument("--movies", type=Path, default=ROOT.parent / "ml-32m/movies.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/d10_mvp" /
                        datetime.now(timezone.utc).strftime("D10-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prefix-events", type=int, default=100_000)
    args = parser.parse_args()
    result = run(args.ratings, args.movies, args.out, device_name=args.device,
                 steps=args.steps, prefix_events=args.prefix_events,
                 fallback_cpu=not args.no_cpu_fallback)
    print(json.dumps({"status": result["status"], "manifest": str(args.out / "run_manifest.json"),
                      "selected_pipeline": result["selected_pipeline"]}, indent=2))
