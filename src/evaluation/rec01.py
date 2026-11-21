"""REC-01 warm-cohort comparison on one frozen train graph and candidate set."""
from __future__ import annotations

from src.evaluation.baselines import popularity_topk
from src.evaluation.temporal import cutoff_cohort, evaluate_topk, score_chunks


def compare_recommenders(embeddings, train_state: dict, train_events: list,
                         future_events: list, *, k: int = 20,
                         chunk_size: int = 1024) -> dict:
    """Evaluate LightGCN and Popularity on identical warm users and train movies.

    The caller owns temporal splitting and checkpoint selection. This function
    accepts either validation or test future events, and never trains or selects
    a checkpoint from the supplied future labels.
    """
    if k != 20:
        raise ValueError("REC-01 primary metric requires k=20")
    registry = train_state["registry"]
    if len(embeddings) != len(registry):
        raise ValueError("Embedding table does not match the train registry")
    if any(first_event_id >= len(train_events) for *_, first_event_id in train_state["edges"]):
        raise ValueError("Future edge detected in train graph")
    users = {raw: node for node, kind, raw, *_ in registry if kind == "user"}
    movies = {raw: node for node, kind, raw, *_ in registry if kind == "movie"}
    truth, seen, cohort = cutoff_cohort(train_events, future_events, registry)
    eligible_users = {raw: users[raw] for raw in truth}
    lightgcn = score_chunks(embeddings, embeddings, eligible_users, movies, seen,
                            k=k, chunk_size=chunk_size)
    popularity = popularity_topk(train_state, eligible_users, movies, seen, k=k)
    if set(lightgcn) != set(popularity) or set(lightgcn) != set(truth):
        raise AssertionError("Methods must use the same warm user cohort")
    return {"cohort": cohort,
            "metrics": {"lightgcn": evaluate_topk(lightgcn, truth, k),
                        "popularity": evaluate_topk(popularity, truth, k)},
            "rankings": {"lightgcn": lightgcn, "popularity": popularity}}
