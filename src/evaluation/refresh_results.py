"""Shared toy quality accounting using the existing Recall/NDCG evaluator."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite

from src.evaluation.refresh_boundary import QualityBundle, stable_hash, validate_history
from src.evaluation.temporal import evaluate_topk


def evaluate_quality(bundle: QualityBundle, rankings: dict[str, dict]) -> dict:
    validate_history(bundle.history)
    if (bundle.e <= bundle.history.b or
            bundle.eligible_label_pairs != tuple(sorted(set(bundle.eligible_label_pairs))) or
            bundle.eligible_user_ids != tuple(sorted({u for u, _ in bundle.eligible_label_pairs})) or
            bundle.eligible_users_sha256 != stable_hash(bundle.eligible_user_ids) or
            bundle.labels_sha256 != stable_hash(bundle.eligible_label_pairs) or
            bundle.unique_positive_pairs != sum(dict(bundle.exclusions).values()) or
            dict(bundle.exclusions)['eligible'] != len(bundle.eligible_label_pairs) or
            bundle.within_window_repeat_events != bundle.positive_events - bundle.unique_positive_pairs or
            bundle.warm_pair_coverage != (len(bundle.eligible_label_pairs) / bundle.unique_positive_pairs
                                          if bundle.unique_positive_pairs else None) or
            bundle.warm_user_coverage != (len(bundle.eligible_user_ids) / bundle.distinct_positive_users
                                          if bundle.distinct_positive_users else None) or
            bundle.digest != stable_hash([bundle.history.digest, bundle.split,
                                          bundle.full_window, bundle.e,
                                          bundle.eligible_user_ids, bundle.eligible_label_pairs,
                                          bundle.exclusions, bundle.rating_events,
                                          bundle.positive_events, bundle.distinct_positive_users,
                                          bundle.no_candidate_users])):
        raise ValueError('Incompatible offline quality bundle')
    if set(rankings) != {'S0', 'S1', 'S2'}:
        raise ValueError('All three strategies are required')
    relevant = defaultdict(set)
    for user, movie in bundle.eligible_label_pairs:
        relevant[user].add(movie)
    catalog = {raw for raw, _ in bundle.history.movie_ids}
    seen = bundle.history.mask.seen_for(bundle.eligible_user_ids)
    per_user, windows = [], []
    for strategy in ('S0', 'S1', 'S2'):
        ranked = rankings[strategy]
        if set(ranked) != set(bundle.eligible_user_ids):
            raise ValueError('Offline users differ across strategies')
        for user in bundle.eligible_user_ids:
            movies = [movie for movie, _ in ranked[user]]
            if (len(movies) != min(20, len(catalog - seen[user])) or
                    len(movies) != len(set(movies)) or
                    not set(movies) <= catalog or set(movies) & seen[user] or
                    any(not isfinite(score) for _, score in ranked[user]) or
                    ranked[user] != sorted(ranked[user], key=lambda item: (-item[1], item[0]))):
                raise ValueError('Invalid or masked Top-20')
            metric = evaluate_topk({user: ranked[user]}, {user: relevant[user]}, 20)
            per_user.append(dict(strategy=strategy, split=bundle.split,
                                 full_window=bundle.full_window,
                                 raw_user_id=user, b=bundle.history.b,
                                 e=bundle.e, bundle_sha256=bundle.digest,
                                 labels_sha256=bundle.labels_sha256,
                                 relevant_count=len(relevant[user]), top20=movies,
                                 recall_20=metric['recall'], ndcg_20=metric['ndcg']))
        metric = evaluate_topk(ranked, relevant, 20)
        windows.append(dict(strategy=strategy, split=bundle.split,
                            full_window=bundle.full_window,
                            b=bundle.history.b, e=bundle.e,
                            bundle_sha256=bundle.digest, eligible_users=metric['users'],
                            eligible_labels=len(bundle.eligible_label_pairs),
                            mean_recall_20=metric['recall'], mean_ndcg_20=metric['ndcg']))
    return dict(per_user=per_user, windows=windows, coverage=coverage(bundle))


def coverage(bundle: QualityBundle) -> dict:
    return dict(split=bundle.split, full_window=bundle.full_window,
                b=bundle.history.b, e=bundle.e, exclusions=dict(bundle.exclusions),
                rating_events=bundle.rating_events, positive_events=bundle.positive_events,
                distinct_positive_users=bundle.distinct_positive_users,
                unique_positive_pairs=bundle.unique_positive_pairs,
                within_window_repeat_events=bundle.within_window_repeat_events,
                eligible_pairs=len(bundle.eligible_label_pairs),
                eligible_users=len(bundle.eligible_user_ids),
                eligible_movies=len({m for _, m in bundle.eligible_label_pairs}),
                warm_pair_coverage=bundle.warm_pair_coverage,
                warm_user_coverage=bundle.warm_user_coverage,
                no_candidate_users=bundle.no_candidate_users)


def aggregate_split(window_rows: list[dict], *, split: str) -> dict:
    if split not in ('development', 'holdout', 'toy'):
        raise ValueError('Unknown split')
    by_strategy = defaultdict(list)
    by_window = defaultdict(list)
    for row in window_rows:
        if row['split'] != split:
            raise ValueError('Window belongs to another split')
        by_strategy[row['strategy']].append(row)
        by_window[(row['b'], row['e'])].append(row)
    for rows in by_window.values():
        if (len(rows) != 3 or {r['strategy'] for r in rows} != {'S0', 'S1', 'S2'} or
                len({(r['bundle_sha256'], r['eligible_users'], r['eligible_labels'],
                      r['full_window']) for r in rows}) != 1):
            raise ValueError('Strategies have incompatible window denominators')
    result = {}
    for strategy in ('S0', 'S1', 'S2'):
        rows = by_strategy[strategy]
        if len({(r['b'], r['e']) for r in rows}) != len(rows):
            raise ValueError('Duplicate windows')
        complete = [r for r in rows if r['full_window']]
        nonempty = [r for r in complete if r['eligible_users']]
        result[strategy] = dict(windows=len(rows), full_windows=len(complete),
                                empty_windows=len(complete) - len(nonempty),
                                macro_recall_20=(sum(r['mean_recall_20'] for r in nonempty) / len(nonempty)
                                                 if nonempty else None),
                                macro_ndcg_20=(sum(r['mean_ndcg_20'] for r in nonempty) / len(nonempty)
                                               if nonempty else None),
                                macro_denominator_windows=len(nonempty))
    bundles = {tuple(sorted((r['b'], r['e'], r['bundle_sha256']) for r in by_strategy[s]))
               for s in ('S0', 'S1', 'S2')}
    if len(bundles) != 1:
        raise ValueError('Strategies do not share windows/bundles')
    return dict(split=split, strategies=result)


def paired_comparisons(per_user_rows: list[dict]) -> dict:
    grouped = defaultdict(dict)
    for row in per_user_rows:
        if not row['full_window']:
            continue
        key = (row['b'], row['e'], row['raw_user_id'], row['bundle_sha256'])
        if row['strategy'] in grouped[key]:
            raise ValueError('Duplicate user/strategy quality row')
        grouped[key][row['strategy']] = row
    by_window = defaultdict(list)
    for (b, e, user, bundle_hash), records in grouped.items():
        if set(records) != {'S0', 'S1', 'S2'}:
            raise ValueError('Unpaired strategy rows')
        if len({(r['split'], r['relevant_count'], r['labels_sha256'])
                for r in records.values()}) != 1:
            raise ValueError('Paired strategies have different labels')
        deltas = {}
        for label, left, right in (('S1-S0', 'S1', 'S0'), ('S2-S0', 'S2', 'S0'),
                                   ('S1-S2', 'S1', 'S2')):
            deltas[label] = {metric: records[left][metric] - records[right][metric]
                             for metric in ('recall_20', 'ndcg_20')}
        by_window[(b, e, bundle_hash)].append((user, deltas))
    windows = []
    for (b, e, bundle_hash), rows in sorted(by_window.items()):
        windows.append(dict(b=b, e=e, bundle_sha256=bundle_hash, users=len(rows),
                            deltas={label: {metric: sum(d[label][metric] for _, d in rows) / len(rows)
                                            for metric in ('recall_20', 'ndcg_20')}
                                    for label in ('S1-S0', 'S2-S0', 'S1-S2')}))
    macro = {label: {metric: (sum(w['deltas'][label][metric] for w in windows) / len(windows)
                             if windows else None)
                     for metric in ('recall_20', 'ndcg_20')}
             for label in ('S1-S0', 'S2-S0', 'S1-S2')}
    return dict(windows=windows, macro=macro, macro_denominator_windows=len(windows),
                distinct_users=len({key[2] for key in grouped}), user_windows=len(grouped))
