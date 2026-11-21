"""Strict event-time split and raw-ID, train-catalog Top-K evaluation."""
from collections import defaultdict
from math import log2


def split_events(events, train_end, valid_end):
    """Cut at timestamps: train <= train_end, valid (train_end, valid_end]."""
    if train_end >= valid_end:
        raise ValueError("Cutoffs must increase")
    ordered = sorted(events, key=lambda e: (e[1], e[0]))
    if [e[0] for e in ordered] != list(range(len(ordered))):
        raise ValueError("Expected contiguous chronological canonical event IDs")
    train = [e for e in ordered if e[1] <= train_end]
    valid = [e for e in ordered if train_end < e[1] <= valid_end]
    test = [e for e in ordered if e[1] > valid_end]
    if not train or not valid or not test:
        raise ValueError("Each temporal partition must be nonempty")
    return train, valid, test


def cutoff_cohort(train, future, registry, threshold=4.0):
    """Future unique positives whose user and movie already exist in train graph."""
    users = {raw: node for node, kind, raw, *_ in registry if kind == "user"}
    movies = {raw: node for node, kind, raw, *_ in registry if kind == "movie"}
    seen = defaultdict(set)
    for _, _, user, movie, rating in train:
        if rating >= threshold and user in users and movie in movies:
            seen[user].add(movie)
    future_pairs = {(user, movie) for _, _, user, movie, rating in future if rating >= threshold}
    relevant = defaultdict(set)
    exclusions = defaultdict(int)
    for user, movie in future_pairs:
        if user not in users:
            exclusions["cold_user"] += 1
        elif movie not in movies:
            exclusions["cold_movie"] += 1
        elif movie in seen[user]:
            exclusions["train_seen_repeat"] += 1
        else:
            relevant[user].add(movie)
    eligible = {user: items for user, items in relevant.items() if len(seen[user]) < len(movies)}
    exclusions["no_candidates_users"] = len(relevant) - len(eligible)
    return eligible, seen, {"future_positive_pairs": len(future_pairs),
                            "warm_users": len(eligible), "warm_pairs": sum(map(len, eligible.values())),
                            "exclusions": dict(exclusions), "catalog_movies": len(movies)}


def score_chunks(user_vectors, movie_vectors, user_ids, movie_ids, seen, k=10, chunk_size=1024):
    """Exact Top-K over train movies; score one movie chunk at a time."""
    if k <= 0 or chunk_size <= 0:
        raise ValueError("k and chunk_size must be positive")
    import torch
    results = {}
    movie_keys = sorted(movie_ids)
    with torch.no_grad():
        for user in sorted(user_ids):
            u = user_vectors[user_ids[user]]
            best = []
            masked = seen.get(user, set())
            for start in range(0, len(movie_keys), chunk_size):
                keys = movie_keys[start:start + chunk_size]
                indices = torch.tensor([movie_ids[key] for key in keys], device=u.device)
                scores = (movie_vectors[indices] @ u).tolist()
                best.extend((float(score), key) for key, score in zip(keys, scores) if key not in masked)
                best.sort(key=lambda pair: (-pair[0], pair[1]))
                del best[k:]
            results[user] = [(movie, score) for score, movie in best]
    return results


def evaluate_topk(topk, relevant, k):
    """Macro Recall@K and NDCG@K on users with warm future positives."""
    if not relevant:
        return {"users": 0, "recall": None, "ndcg": None}
    recalls, ndcgs = [], []
    for user, truth in relevant.items():
        ranked = [movie for movie, _ in topk[user][:k]]
        hits = [int(movie in truth) for movie in ranked]
        recalls.append(sum(hits) / len(truth))
        dcg = sum(hit / log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / log2(rank + 2) for rank in range(min(k, len(truth))))
        ndcgs.append(dcg / ideal)
    return {"users": len(relevant), "recall": sum(recalls) / len(recalls),
            "ndcg": sum(ndcgs) / len(ndcgs)}
