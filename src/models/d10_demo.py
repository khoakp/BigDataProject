"""Inspect a completed D10 MVP run without retraining or changing artifacts."""
import argparse
import json
from pathlib import Path

from src.graph.full_builder import file_sha256


def show(run_dir: Path, *, check_hashes=True, user_raw_id=None):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise ValueError("MVP run is not PASS")
    if check_hashes:
        for relative, record in manifest["artifacts"].items():
            if file_sha256(run_dir / relative) != record["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {relative}")
    pipeline = run_dir / manifest["selected_pipeline"]
    report = json.loads((pipeline / "integration_report.json").read_text(encoding="utf-8"))
    rows = json.loads((pipeline / "topk_enriched.json").read_text(encoding="utf-8"))
    if user_raw_id is None:
        user_raw_id = next(row["user_raw_id"] for row in rows if row["split"] == "valid")
    chosen = [row for row in rows if row["split"] == "valid" and row["user_raw_id"] == user_raw_id]
    if not chosen:
        raise ValueError("User is not in the validation warm cohort")
    print(f"Run: {manifest['run_id']} | graph/checkpoint version: {report['model_graph_version']}")
    print(f"Train nodes: {manifest['model']['node_count']} | metadata rows: {report['topk_coverage']['metadata_found']}/{report['topk_coverage']['rows']}")
    print(f"Validation Recall@10: {report['metrics']['valid']['recall']:.5f} | NDCG@10: {report['metrics']['valid']['ndcg']:.5f}")
    print(f"User {user_raw_id} - validation Top-10")
    for row in chosen:
        print(f"{row['rank']:2d}. movieId={row['movie_raw_id']} score={row['score']:.5f} | {row['title']} | {row['genres']}")
    return chosen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--user", type=int)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()
    show(args.run_dir, check_hashes=not args.skip_hash_check, user_raw_id=args.user)
