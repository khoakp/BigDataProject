"""Pre-refactor evidence only; run from repo root, never writes old artifacts."""
import csv
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import duckdb
import torch

from src.graph.full_builder import file_sha256, rebuild
from src.graph.incremental import IncrementalGraph
from src.ingest.d02_profile import run as canonicalize
from src.models.rec01_inference import load_rec01_inference
from src.models.rec01_train import Protocol, run


ROOT = Path.cwd()
OUT = ROOT / "results/goal035_baseline"


def write(name, value):
    if (OUT / name).exists():
        raise FileExistsError(f"Baseline must stay immutable: {OUT / name}")
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")


def main():
    torch.set_num_threads(1)
    artifact = ROOT / "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
    tracked = [artifact / name for name in ("report.json", "model_ids.json", "output_manifest.json")]
    tracked += sorted(artifact.glob("*.pt")) + sorted((artifact / "model_graph").glob("*"))
    before = {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in tracked if p.is_file()}
    service = load_rec01_inference(artifact, ROOT.parent / "ml-32m/movies.csv")
    warm = sorted(service.user_nodes)
    users = sorted({warm[0], warm[len(warm)//2], warm[-1], 25062})
    users.append(max(warm) + 1)
    assert users[-1] not in service.user_nodes
    output = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "device": "cpu", "threads": 1, "artifact_hashes": before,
              "graph_version": service.graph_version, "checkpoint_step": service.checkpoint_step,
              "selection": "first/middle/last sorted warm IDs, prior smoke user 25062, max warm ID + 1 cold",
              "comparison_contract": {"movie_ids": "exact ordered equality", "scores_atol": 1e-6,
                                      "scores_rtol": 1e-6},
              "recommendations": [service.recommend(u) for u in users]}
    assert before == {name: file_sha256(ROOT / name) for name in before}
    write("rec01-before.json", output)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "ratings.csv"
        rows = [(99, i+1, 5., i+1) for i in range(22)]
        rows += [(7, 1, 5., 23), (7, 2, 5., 24)]
        rows += [(7, i-21, 5., i+1) for i in range(24, 30)]
        with raw.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("userId", "movieId", "rating", "timestamp"))
            writer.writerows(rows)
        canonicalize(raw, root / "canonical", max_prefix=30)
        events = root / "canonical/canonical_30.parquet"
        protocol = Protocol(source_sha256=file_sha256(events), events=30, distinct_timestamps=30,
                            train_end_ts=24, valid_end_ts=27, train_events=24, valid_events=3,
                            test_events=3, checkpoint_steps=(1, 2), batch_size=4, dim=4, layers=1,
                            graph_batch_events=10, chunk_size=4)
        report = run(events, root / "training", device_name="cpu", protocol=protocol)
        movies = root / "movies.csv"
        movies.write_text("movieId,title,genres\n" + "".join(
            f"{i},Movie {i},Drama\n" for i in range(1, 23)), encoding="utf-8")
        result = load_rec01_inference(root / "training", movies).recommend(7)
        assert report["status"] == "PASS" and report["graph"]["full_equivalence"]["pass"]
        assert len(result["items"]) == 20
        write("smoke-before.json", {"status": "PASS", "pipeline":
              "raw CSV -> canonical events -> full/incremental equivalence -> snapshot export/load -> two BPR steps -> validation selection -> held-out evaluation -> inference",
              "graph": report["graph"], "selected_checkpoint": report["selected_checkpoint"],
              "recommendation": result})

        malformed = root / "malformed.parquet"
        with duckdb.connect() as con:
            con.execute("CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,"
                        "movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
            con.executemany("INSERT INTO e VALUES (?,?,?,?,?,?)", [
                (0, 1, 1, 10, 4., 0), (1, 2, 1, 11, 4., 1),
                (1, 3, 2, 10, 4., 2), (3, 4, 2, 11, 4., 3)])
            con.execute("COPY e TO ? (FORMAT PARQUET)", [str(malformed)])
        full = rebuild(malformed)
        with IncrementalGraph() as graph:
            try:
                graph.apply_batch(0, malformed)
            except ValueError as exc:
                rejection = str(exc)
            else:
                raise AssertionError("Expected incremental rejection")
        raw.write_text("userId,movieId,rating,timestamp\n0,10,4,1\n", encoding="utf-8")
        profile = canonicalize(raw, root / "invalid", max_prefix=1)
        assert profile["rows"] == 1
        write("audit-reproductions.json", {
            "duplicate_interior_event_id": {"input_ids": [0, 1, 1, 3], "full_accepted": full,
                                             "incremental_rejected": rejection},
            "nonpositive_raw_user": {"raw_user_id": 0, "canonicalizer_accepted": True}})
    print(json.dumps({"checkpoint": "PASS", "users": users, "smoke": "PASS", "reproductions": 2}))


if __name__ == "__main__":
    main()
