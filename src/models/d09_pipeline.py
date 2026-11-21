"""D09 end-to-end canonical events -> LightGCN -> raw and metadata Top-K."""
import argparse
import json
from pathlib import Path

import torch

from src.graph.full_builder import file_sha256
from src.metadata.movie_lookup import enrich_topk, load_movies
from src.models.d08_baseline import ROOT, run as train_and_evaluate


def run(source: Path, movies_source: Path, output: Path, *, device_name: str, steps=100):
    output.mkdir(parents=True, exist_ok=True)
    # D08 executes the complete train-only graph, normalization, model and raw ranking path.
    report = train_and_evaluate(source, output, device_name=device_name, steps=steps,
                                graph_method="incremental")
    raw = json.loads((output / "topk_raw.json").read_text(encoding="utf-8"))
    movies, movies_profile = load_movies(movies_source)
    enriched, coverage = enrich_topk(raw, movies)
    identity = ("split", "user_raw_id", "movie_raw_id", "rank", "score")
    if len(enriched) != len(raw) or any(tuple(a[key] for key in identity) !=
                                       tuple(b[key] for key in identity) for a, b in zip(raw, enriched)):
        raise AssertionError("Metadata enrichment changed raw ranking")
    # Evaluation is upstream of metadata; retain its exact result and prove that
    # no metadata field is consumed when recomputing from either row set.
    from src.evaluation.temporal import evaluate_topk
    from src.models.d08_baseline import load_events, choose_cutoffs
    from src.evaluation.temporal import split_events, cutoff_cohort
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    events = load_events(source)
    train_end, valid_end = choose_cutoffs(events)
    train, valid, test = split_events(events, train_end, valid_end)
    for split, future in (("valid", valid), ("test", test)):
        truth, _, _ = cutoff_cohort(train, future, checkpoint["registry"])
        def ranking(rows):
            topk = {}
            for row in rows:
                if row["split"] == split:
                    topk.setdefault(row["user_raw_id"], []).append((row["movie_raw_id"], row["score"]))
            return topk
        original = evaluate_topk(ranking(raw), truth, report["config"]["k"])
        after = evaluate_topk(ranking(enriched), truth, report["config"]["k"])
        if original != after or original != report["metrics"][split]:
            raise AssertionError("Metadata enrichment changed metrics")
    version = report["graph"]["version"]
    for rows in (raw, enriched):
        for row in rows:
            row.update(canonical_version=version, model_graph_version=version,
                       model_checkpoint_version=version)
    (output / "topk_raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (output / "topk_enriched.json").write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    integration = {"status": "PASS", "source_sha256": report["source_sha256"],
                   "movies_source_sha256": file_sha256(movies_source),
                   "checkpoint_sha256": report["checkpoint_sha256"],
                   "canonical_version": version, "model_graph_version": version,
                   "model_checkpoint_version": version,
                   "movies_profile": movies_profile, "topk_coverage": coverage,
                   "raw_rows_equal_enriched_rows": True, "ranking_fields_unchanged": True,
                   "metrics_unchanged": True, "metrics": report["metrics"]}
    (output / "integration_report.json").write_text(json.dumps(integration, indent=2), encoding="utf-8")
    return integration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/canonical_events/canonical_100000.parquet")
    parser.add_argument("--movies", type=Path, default=ROOT.parent / "ml-32m/movies.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/d09_pipeline")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.movies, args.out, device_name=args.device, steps=args.steps), indent=2))
