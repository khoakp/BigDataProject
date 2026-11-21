"""REC-01 1M feedback integration probe with an independent Full Rebuild."""
from __future__ import annotations

import argparse
import json
import platform
import time
import uuid
from pathlib import Path

import duckdb

from src.app.feedback import FeedbackStore
from src.app.rec01_view import train_rated_ids
from src.graph.equivalence import compare_states, full_state, incremental_state
from src.graph.full_builder import file_sha256, rebuild
from src.graph.incremental import IncrementalGraph
from src.models.rec01_inference import load_rec01_inference


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
EVENTS = ROOT / "data/processed/canonical_events/canonical_1000000.parquet"
MOVIES = ROOT.parent / "ml-32m/movies.csv"


def _combined_train(source: Path, batch: Path, train_end_ts: int, target: Path) -> None:
    with duckdb.connect() as con:
        con.execute("SET threads=1")
        con.execute("CREATE TEMP TABLE all_events AS "
                    "SELECT event_id,event_ts,user_raw_id,movie_raw_id,rating,source_row_id "
                    "FROM read_parquet(?) WHERE event_ts <= ? UNION ALL "
                    "SELECT event_id,event_ts,user_raw_id,movie_raw_id,rating,source_row_id "
                    "FROM read_parquet(?)", [str(source), train_end_ts, str(batch)])
        escaped = str(target.resolve()).replace("\\", "/").replace("'", "''")
        con.execute(f"COPY (SELECT * FROM all_events ORDER BY event_id) "
                    f"TO '{escaped}' (FORMAT PARQUET)")


def run_probe(run: Path, source: Path, movies: Path, output: Path,
              *, user_id: int = 173074) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original_graph = run / "graph_build/graph.duckdb"
    original_hash = file_sha256(original_graph)
    started = time.perf_counter()
    model = load_rec01_inference(run, movies)
    store = FeedbackStore(run, source, output / "feedback")
    setup_seconds = time.perf_counter() - started
    if user_id not in model.user_nodes:
        raise ValueError("Probe user must have a train embedding")
    rated = train_rated_ids(source, user_id, model.report["protocol"]["train_end_ts"])
    choices = [movie for movie in sorted(model.movie_nodes) if movie not in rated][:20]
    if len(choices) != 20:
        raise ValueError("Probe user lacks 20 unrated train movies")
    ratings = [{"user_raw_id": user_id, "movie_raw_id": movie,
                "rating": 5.0 if index % 2 == 0 else 2.5}
               for index, movie in enumerate(choices)]
    before = model.recommend(user_id)
    cold_before = model.recommend(max(model.user_nodes) + 1)
    base = store.status()
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    applied = store.apply_ratings(ratings, request_id)
    apply_seconds = time.perf_counter() - started
    current = store.status()
    reopened = FeedbackStore(run, source, output / "feedback")
    replay = reopened.apply_ratings(ratings, request_id)
    if (applied["status"] != "APPLIED" or replay["status"] != "NO_OP" or
            applied["inserted_edges"] != 10 or
            current["positive_edges"] != base["positive_edges"] + 10 or
            current["canonical_graph_version"] != base["canonical_graph_version"] + 1 or
            current["model_graph_version"] != base["model_graph_version"] or
            current["new_feedback_events"] != 20 or
            reopened.status() != current or
            model.recommend(user_id) != before or
            model.recommend(max(model.user_nodes) + 1) != cold_before):
        raise AssertionError("Feedback/replay/version/inference invariant failed")

    started = time.perf_counter()
    batch = reopened.batches_dir / f"batch_{applied['batch_id']:05d}.parquet"
    combined = output / "combined_train_plus_feedback.parquet"
    _combined_train(source, batch, model.report["protocol"]["train_end_ts"], combined)
    full = rebuild(combined, output / "full_reference")
    with IncrementalGraph(reopened.graph_path) as graph:
        comparison = compare_states(full_state(output / "full_reference"),
                                    incremental_state(graph))
        snapshots = graph.con.execute(
            "SELECT model_graph_version FROM model_graph_snapshots ORDER BY 1"
        ).fetchall()
    reference_seconds = time.perf_counter() - started
    if not comparison["pass"] or full["edges"] != current["positive_edges"]:
        raise AssertionError("Full/Incremental feedback state differs")
    if max(version for (version,) in snapshots) != base["model_graph_version"]:
        raise AssertionError("Feedback unexpectedly materialized a model graph")
    if file_sha256(original_graph) != original_hash:
        raise AssertionError("Sealed Kaggle graph was modified")

    report = {
        "status": "PASS", "run_id": model.report["run_id"],
        "python": platform.python_version(), "duckdb": duckdb.__version__,
        "source_sha256": file_sha256(source),
        "sealed_graph_sha256": original_hash,
        "combined_events_sha256": file_sha256(combined),
        "user_raw_id": user_id, "feedback_ratings": ratings,
        "request_id": request_id, "base": base, "apply": applied,
        "after": current, "replay": replay, "full_rebuild": full,
        "exact_comparison": comparison,
        "model_snapshot_versions": [version for (version,) in snapshots],
        "recommendations_unchanged": True,
        "seconds_single_run": {"load_and_copy": setup_seconds,
                               "apply_batch": apply_seconds,
                               "full_reference_and_compare": reference_seconds},
        "limits": "One 20-rating batch on the 1M REC-01 train graph; wall times are diagnostic, not an EXP-01 benchmark",
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--source", type=Path, default=EVENTS)
    parser.add_argument("--movies", type=Path, default=MOVIES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outcome = run_probe(args.run, args.source, args.movies, args.out)
    print(json.dumps({"status": outcome["status"],
                      "graph_version": outcome["after"]["canonical_graph_version"],
                      "model_graph_version": outcome["after"]["model_graph_version"],
                      "full_equivalence": outcome["exact_comparison"]["pass"]}))
