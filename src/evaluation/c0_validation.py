"""Fixed-catalog, half-open monthly validation for a frozen G0 embedding table."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from src.evaluation.temporal import evaluate_topk, score_chunks


START = 1577836800
END = 1609459200


def score_batched(embeddings, user_ids, movie_ids, seen, *, k=20, batch_size=32):
    """Exact fixed-catalog Top-K with stable raw-movie-ID tie breaking."""
    import torch

    if k <= 0 or batch_size <= 0:
        raise ValueError('k and batch_size must be positive')
    movie_keys = sorted(movie_ids)
    if not movie_keys:
        raise ValueError('Empty fixed movie catalog')
    movie_columns = {movie: column for column, movie in enumerate(movie_keys)}
    movie_vectors = embeddings[torch.tensor([movie_ids[m] for m in movie_keys],
                                             device=embeddings.device)]
    if not torch.isfinite(movie_vectors).all():
        raise ValueError('Nonfinite propagated movie vectors')
    output = {}
    with torch.inference_mode():
        keys = sorted(user_ids)
        for first in range(0, len(keys), batch_size):
            batch = keys[first:first + batch_size]
            vectors = embeddings[torch.tensor([user_ids[u] for u in batch],
                                               device=embeddings.device)]
            if not torch.isfinite(vectors).all():
                raise ValueError('Nonfinite propagated user vectors')
            scores = vectors @ movie_vectors.T
            for row, user in enumerate(batch):
                masked = [movie_columns[m] for m in seen.get(user, ()) if m in movie_columns]
                if masked:
                    scores[row, masked] = -torch.inf
            # Stable order preserves ascending raw movie ID when scores tie.
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            values = scores.gather(1, order)
            for row, user in enumerate(batch):
                output[user] = [(movie_keys[column], float(score))
                                for column, score in zip(order[row].tolist(), values[row].tolist())
                                if score != -torch.inf]
    return output


def boundaries():
    return [int(datetime(2020 + month // 12, month % 12 + 1, 1,
                         tzinfo=timezone.utc).timestamp()) for month in range(13)]


def validation_windows(events, registry, base_edges, *, start=START, end=END,
                       window_boundaries=None):
    """Events are (event_id, ts, user_raw, movie_raw, rating, source_row_id).

    The registry and base edges define model IDs. Future labels never enter the
    model; each mask uses positive pairs strictly before its own boundary.
    """
    cuts = window_boundaries or boundaries()
    if cuts[0] != start or cuts[-1] != end or any(a >= b for a, b in zip(cuts, cuts[1:])):
        raise ValueError('Invalid validation boundaries')
    users = {raw: node for node, kind, raw, *_ in registry if kind == 'user'}
    movies = {raw: node for node, kind, raw, *_ in registry if kind == 'movie'}
    reverse_users = {node: raw for raw, node in users.items()}
    reverse_movies = {node: raw for raw, node in movies.items()}
    if len(users) + len(movies) != len(registry):
        raise ValueError('Invalid fixed mapping')
    seen = defaultdict(set)
    for user_node, movie_node, *_ in base_edges:
        if user_node not in reverse_users or movie_node not in reverse_movies:
            raise ValueError('Base edge outside fixed mapping')
        seen[reverse_users[user_node]].add(reverse_movies[movie_node])
    ordered = sorted(events, key=lambda row: row[0])
    if any(len(row) != 6 for row in ordered) or any(
        left[0] >= right[0] or left[1:] and
        (left[1], left[2], left[3], left[5]) >=
        (right[1], right[2], right[3], right[5])
        for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Events are not in canonical order')
    if any(not start <= row[1] < end for row in ordered):
        raise ValueError('Validation input contains an event outside 2020')
    result = []
    cursor = 0
    for boundary, next_boundary in zip(cuts, cuts[1:]):
        while cursor < len(ordered) and ordered[cursor][1] < boundary:
            _, _, user, movie, rating, _ = ordered[cursor]
            if rating >= 4 and user in users and movie in movies:
                seen[user].add(movie)
            cursor += 1
        frozen = {user: set(items) for user, items in seen.items()}
        labels = defaultdict(set)
        counts = dict(cold_user=0, cold_movie=0, seen=0, eligible=0)
        stop = cursor
        pairs = set()
        while stop < len(ordered) and ordered[stop][1] < next_boundary:
            _, _, user, movie, rating, _ = ordered[stop]
            if rating >= 4:
                pairs.add((user, movie))
            stop += 1
        for user, movie in sorted(pairs):
            if user not in users:
                counts['cold_user'] += 1
            elif movie not in movies:
                counts['cold_movie'] += 1
            elif movie in frozen.get(user, ()):
                counts['seen'] += 1
            else:
                labels[user].add(movie)
                counts['eligible'] += 1
        result.append(dict(start=boundary, end=next_boundary, relevant=dict(labels),
                           seen=frozen, counts=counts))
    return result


def evaluate_embeddings(embeddings, registry, windows, *, selected=None, k=20,
                        batch_size=None):
    """Partition windows without changing users, ranking, or aggregation."""
    users = {raw: node for node, kind, raw, *_ in registry if kind == 'user'}
    movies = {raw: node for node, kind, raw, *_ in registry if kind == 'movie'}
    if selected is None:
        selected = tuple(range(len(windows)))
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < len(windows) for i in selected):
        raise ValueError('Invalid window partition')
    rows = []
    for index in selected:
        window = windows[index]
        eligible = window['relevant']
        if not eligible:
            raise ValueError('Validation window has no eligible users')
        if batch_size is None:
            ranking = score_chunks(embeddings, embeddings,
                                   {u: users[u] for u in eligible}, movies,
                                   window['seen'], k=k)
        else:
            ranking = score_batched(embeddings, {u: users[u] for u in eligible},
                                    movies, window['seen'], k=k, batch_size=batch_size)
        metric = evaluate_topk(ranking, eligible, k)
        rows.append(dict(index=index, start=window['start'], end=window['end'],
                         users=metric['users'], labels=sum(map(len, eligible.values())),
                         recall_20=metric['recall'], ndcg_20=metric['ndcg']))
    return rows


def aggregate_validation(parts, expected_windows=12):
    """Accept disjoint worker outputs only after all windows are present once."""
    rows = [row for part in parts for row in part]
    if len(rows) != expected_windows or {r['index'] for r in rows} != set(range(expected_windows)):
        raise ValueError('Incomplete or duplicate validation windows')
    rows.sort(key=lambda row: row['index'])
    return dict(windows=rows, validation_recall_20=sum(r['recall_20'] for r in rows)/len(rows),
                validation_ndcg_20=sum(r['ndcg_20'] for r in rows)/len(rows))
