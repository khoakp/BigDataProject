"""Train-only recommendation baselines using the temporal evaluator's ID contract."""
from __future__ import annotations

from collections import defaultdict


def popularity_topk(state: dict, user_ids: dict[int, int], movie_ids: dict[int, int],
                    seen: dict[int, set[int]], k: int = 20) -> dict[int, list[tuple[int, float]]]:
    """Rank train-catalog movies by unique positive train users.

    ``state`` is the train-only graph state with ``registry`` and ``edges`` rows.
    ``user_ids``, ``movie_ids`` and ``seen`` have the same raw-ID contract as
    ``temporal.score_chunks``. The caller supplies the eligible user cohort and
    candidate catalog; future events must never enter ``state``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    registry = {(kind, raw_id): node_id for node_id, kind, raw_id, *_ in state["registry"]}
    for kind, mapping in (("user", user_ids), ("movie", movie_ids)):
        if any(registry.get((kind, raw_id)) != node_id for raw_id, node_id in mapping.items()):
            raise ValueError(f"{kind} mapping is not in the train graph registry")
    train_movies = {raw_id for kind, raw_id in registry if kind == "movie"}
    if set(movie_ids) != train_movies:
        raise ValueError("Movie candidates must equal the complete train graph catalog")

    users_by_movie = defaultdict(set)
    for user_node, movie_node, *_ in state["edges"]:
        users_by_movie[movie_node].add(user_node)
    ordered_movies = sorted(movie_ids, key=lambda raw_id: (-len(users_by_movie[movie_ids[raw_id]]), raw_id))
    result = {}
    for user in sorted(user_ids):
        masked = seen.get(user, set())
        result[user] = [(movie, float(len(users_by_movie[movie_ids[movie]])))
                        for movie in ordered_movies if movie not in masked][:k]
    return result
