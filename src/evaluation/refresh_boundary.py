"""Frozen pre-boundary projection and separate evaluator-only future labels."""
from __future__ import annotations

import hashlib
import json
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.artifacts import file_sha256
from src.graph.model_replay import ModelGraphReplay, load_projection_snapshot


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class MaskRef:
    snapshot: Path
    manifest_sha256: str
    mapping_sha256: str
    source_sha256: str
    version: int
    boundary: int
    pairs_sha256: str

    def seen_for(self, raw_user_ids: tuple[int, ...]) -> dict[int, set[int]]:
        if file_sha256(self.snapshot / 'manifest.json') != self.manifest_sha256:
            raise ValueError('Observed mask snapshot changed')
        load_projection_snapshot(self.snapshot, expected_mapping=self.mapping_sha256,
                                 expected_version=self.version, expected_boundary=self.boundary,
                                 expected_source=self.source_sha256, materialize=False)
        if not raw_user_ids:
            return {}
        with duckdb.connect() as con:
            rows = con.execute('''SELECT u.raw_id,m.raw_id FROM read_parquet(?) e
                JOIN read_parquet(?) u ON e.user_node_id=u.node_id
                JOIN read_parquet(?) m ON e.movie_node_id=m.node_id
                WHERE u.raw_id IN (SELECT unnest(?::BIGINT[])) ORDER BY 1,2''',
                [str(self.snapshot / 'edges.parquet'), str(self.snapshot / 'nodes.parquet'),
                 str(self.snapshot / 'nodes.parquet'), list(raw_user_ids)]).fetchall()
        seen = {u: set() for u in raw_user_ids}
        for user, movie in rows:
            seen[user].add(movie)
        return seen


@dataclass(frozen=True)
class BoundaryHistory:
    b: int
    source_sha256: str
    cohort_sha256: str
    mapping_sha256: str
    projection_version: int
    canonical_watermark_event_id: int
    last_canonical_order_key: tuple[int, int, int, int]
    snapshot: Path
    snapshot_sha256: str
    edges_sha256: str
    degrees_sha256: str
    coo_sha256: str
    user_ids: tuple[tuple[int, int], ...]
    movie_ids: tuple[tuple[int, int], ...]
    catalog_sha256: str
    mask: MaskRef
    digest: str


@dataclass(frozen=True)
class RankRequest:
    history: BoundaryHistory
    raw_user_ids: tuple[int, ...]
    workload: str
    k: int = 20

    def __post_init__(self):
        if (self.k != 20 or self.workload not in ('offline_quality', 'serving_like') or
                self.raw_user_ids != tuple(sorted(set(self.raw_user_ids))) or
                not set(self.raw_user_ids) <= {raw for raw, _ in self.history.user_ids}):
            raise ValueError('Invalid common ranking request')


@dataclass(frozen=True)
class QualityBundle:
    history: BoundaryHistory
    split: str
    full_window: bool
    e: int
    eligible_user_ids: tuple[int, ...]
    eligible_label_pairs: tuple[tuple[int, int], ...]
    eligible_users_sha256: str
    labels_sha256: str
    exclusions: tuple[tuple[str, int], ...]
    rating_events: int
    positive_events: int
    distinct_positive_users: int
    unique_positive_pairs: int
    within_window_repeat_events: int
    warm_pair_coverage: float | None
    warm_user_coverage: float | None
    no_candidate_users: int
    digest: str


@dataclass(frozen=True)
class BenchmarkUsers:
    eligible_pool_count: int
    selected: tuple[int, ...]
    selected_sha256: str
    mask_sha256: str
    status: str


def _mask_hash(snapshot: Path) -> str:
    """Hash sorted raw positive pairs in bounded batches, without a 5M-pair bundle."""
    digest = hashlib.sha256(b'[')
    first = True
    with duckdb.connect() as con:
        cursor = con.execute('''SELECT u.raw_id,m.raw_id FROM read_parquet(?) e
            JOIN read_parquet(?) u ON e.user_node_id=u.node_id
            JOIN read_parquet(?) m ON e.movie_node_id=m.node_id ORDER BY 1,2''',
            [str(snapshot / 'edges.parquet'), str(snapshot / 'nodes.parquet'),
             str(snapshot / 'nodes.parquet')])
        while rows := cursor.fetchmany(65536):
            for row in rows:
                if not first:
                    digest.update(b',')
                digest.update(json.dumps(row, separators=(',', ':')).encode('utf-8'))
                first = False
    digest.update(b']')
    return digest.hexdigest()


def freeze_boundary(replay: ModelGraphReplay, b: int, snapshot: Path,
                    *, profile_timings: dict | None = None) -> BoundaryHistory:
    if type(b) is not int:
        raise ValueError('UTC boundary must be integer seconds')
    tick = perf_counter()
    version = replay.advance(b)
    if profile_timings is not None:
        profile_timings['canonical_replay_s'] = perf_counter() - tick
    if version['observed_before_ts'] != b:
        raise ValueError('Replay did not freeze requested boundary')
    tick = perf_counter()
    replay.export_snapshot(snapshot, expected_version=version['projection_version'])
    if profile_timings is not None:
        profile_timings['snapshot_export_s'] = perf_counter() - tick
    manifest, _, _ = load_projection_snapshot(
        snapshot, expected_mapping=replay.meta['mapping_sha256'],
        expected_version=version['projection_version'], expected_boundary=b,
        expected_source=replay.meta['source_sha256'], materialize=False,
        profile_timings=profile_timings)
    if manifest['last_canonical_order_key'][0] >= b:
        raise ValueError('Future event in projection')
    with duckdb.connect() as con:
        rows = con.execute('SELECT node_id,node_type,raw_id FROM read_parquet(?) ORDER BY node_id',
                           [str(snapshot / 'nodes.parquet')]).fetchall()
    users = tuple(sorted((raw, node) for node, kind, raw in rows if kind == 'user'))
    movies = tuple(sorted((raw, node) for node, kind, raw in rows if kind == 'movie'))
    if len(users) + len(movies) != len(rows):
        raise ValueError('Invalid fixed mapping')
    manifest_sha = file_sha256(snapshot / 'manifest.json')
    tick = perf_counter()
    mask_sha = _mask_hash(snapshot)
    if profile_timings is not None:
        profile_timings['mask_hash_s'] = perf_counter() - tick
    mask = MaskRef(snapshot, manifest_sha, manifest['mapping_sha256'],
                   manifest['source_sha256'], manifest['projection_version'], b, mask_sha)
    fields = [b, manifest['source_sha256'], manifest['cohort_manifest_sha256'],
              manifest['mapping_sha256'], manifest['projection_version'],
              manifest['canonical_watermark_event_id'], manifest['last_canonical_order_key'],
              manifest_sha, manifest['edges_sha256'], manifest['degrees_sha256'],
              manifest['coo_sha256'], users, movies, mask_sha]
    return BoundaryHistory(b, manifest['source_sha256'], manifest['cohort_manifest_sha256'],
                           manifest['mapping_sha256'], manifest['projection_version'],
                           manifest['canonical_watermark_event_id'],
                           tuple(manifest['last_canonical_order_key']), snapshot, manifest_sha,
                           manifest['edges_sha256'], manifest['degrees_sha256'],
                           manifest['coo_sha256'], users, movies,
                           stable_hash([raw for raw, _ in movies]), mask, stable_hash(fields))


def validate_history(history: BoundaryHistory) -> None:
    if type(history) is not BoundaryHistory:
        raise TypeError('Strategy input must be pre-boundary history')
    if file_sha256(history.snapshot / 'manifest.json') != history.snapshot_sha256:
        raise ValueError('Frozen snapshot manifest changed')
    manifest, _, _ = load_projection_snapshot(
        history.snapshot, expected_mapping=history.mapping_sha256,
        expected_version=history.projection_version,
        expected_boundary=history.b, expected_source=history.source_sha256,
        materialize=False)
    with duckdb.connect() as con:
        rows = con.execute('SELECT node_id,node_type,raw_id FROM read_parquet(?) ORDER BY node_id',
                           [str(history.snapshot / 'nodes.parquet')]).fetchall()
    users = tuple(sorted((raw, node) for node, kind, raw in rows if kind == 'user'))
    movies = tuple(sorted((raw, node) for node, kind, raw in rows if kind == 'movie'))
    mask_sha = _mask_hash(history.snapshot)
    fields = [history.b, manifest['source_sha256'], manifest['cohort_manifest_sha256'],
              manifest['mapping_sha256'], manifest['projection_version'],
              manifest['canonical_watermark_event_id'], manifest['last_canonical_order_key'],
              history.snapshot_sha256, manifest['edges_sha256'], manifest['degrees_sha256'],
              manifest['coo_sha256'], users, movies, mask_sha]
    if (history.cohort_sha256 != manifest['cohort_manifest_sha256'] or
            history.canonical_watermark_event_id != manifest['canonical_watermark_event_id'] or
            history.last_canonical_order_key != tuple(manifest['last_canonical_order_key']) or
            history.edges_sha256 != manifest['edges_sha256'] or
            history.degrees_sha256 != manifest['degrees_sha256'] or
            history.coo_sha256 != manifest['coo_sha256'] or
            history.user_ids != users or history.movie_ids != movies or
            history.catalog_sha256 != stable_hash([raw for raw, _ in movies]) or
            history.mask != MaskRef(history.snapshot, history.snapshot_sha256,
                                    history.mapping_sha256, history.source_sha256,
                                    history.projection_version, history.b, mask_sha) or
            history.digest != stable_hash(fields)):
        raise ValueError('Frozen history fields differ from verified projection')


def _check_history(replay: ModelGraphReplay, history: BoundaryHistory) -> None:
    validate_history(history)
    current = replay.current()
    if (file_sha256(Path(replay.meta['source_path'])) != history.source_sha256 or
            current['observed_before_ts'] != history.b or
            current['projection_version'] != history.projection_version or
            current['source_sha256'] != history.source_sha256 or
            current['mapping_sha256'] != history.mapping_sha256 or
            current['edges_sha256'] != history.edges_sha256 or
            file_sha256(history.snapshot / 'manifest.json') != history.snapshot_sha256):
        raise ValueError('Incompatible frozen history')


def select_serving_users(replay: ModelGraphReplay, history: BoundaryHistory,
                         limit: int = 1000) -> BenchmarkUsers:
    _check_history(replay, history)
    if limit != 1000:
        raise ValueError('Approved serving-like limit is 1000')
    pool = replay.con.execute('''SELECT DISTINCT e.user_raw_id FROM events e
        JOIN nodes u ON u.node_type='user' AND u.raw_id=e.user_raw_id
        JOIN nodes m ON m.node_type='movie' AND m.raw_id=e.movie_raw_id
        WHERE e.event_ts>=? AND e.event_ts<? AND e.rating>=4
        AND (SELECT count(*) FROM edges x WHERE x.user_node_id=u.node_id) < ?
        ORDER BY 1''', [history.b - 90 * 86400, history.b, len(history.movie_ids)]).fetchall()
    selected = tuple(row[0] for row in pool[:1000])
    return BenchmarkUsers(len(pool), selected, stable_hash(selected), history.mask.pairs_sha256,
                          'OK' if selected else 'NOT_APPLICABLE')


def make_quality_bundle(replay: ModelGraphReplay, history: BoundaryHistory,
                        e: int, *, split: str = 'toy', full_window: bool = True) -> QualityBundle:
    """Evaluator-only path: future labels never enter BoundaryHistory."""
    _check_history(replay, history)
    if (type(e) is not int or e <= history.b or
            split not in ('toy', 'development', 'holdout') or type(full_window) is not bool):
        raise ValueError('Invalid half-open window')
    rating_events, positive_events, positive_users = replay.con.execute('''SELECT count(*),
        count(*) FILTER(WHERE rating>=4),count(DISTINCT user_raw_id) FILTER(WHERE rating>=4)
        FROM events WHERE event_ts>=? AND event_ts<?''', [history.b, e]).fetchone()
    rows = replay.con.execute('''WITH pairs AS (
            SELECT user_raw_id,movie_raw_id FROM events
            WHERE event_ts>=? AND event_ts<? AND rating>=4 GROUP BY 1,2)
        SELECT p.user_raw_id,p.movie_raw_id,
            CASE WHEN u.node_id IS NULL AND bu.user_raw_id IS NULL THEN 'cold_user'
                 WHEN u.node_id IS NULL THEN 'omitted_user'
                 WHEN m.node_id IS NULL AND bm.movie_raw_id IS NULL THEN 'cold_movie'
                 WHEN m.node_id IS NULL THEN 'omitted_movie'
                 WHEN x.first_event_id IS NOT NULL THEN 'seen'
                 ELSE 'eligible' END category
        FROM pairs p
        LEFT JOIN nodes u ON u.node_type='user' AND u.raw_id=p.user_raw_id
        LEFT JOIN nodes m ON m.node_type='movie' AND m.raw_id=p.movie_raw_id
        LEFT JOIN base_users bu ON bu.user_raw_id=p.user_raw_id
        LEFT JOIN base_movies bm ON bm.movie_raw_id=p.movie_raw_id
        LEFT JOIN edges x ON x.user_node_id=u.node_id AND x.movie_node_id=m.node_id
        ORDER BY p.user_raw_id,p.movie_raw_id''', [history.b, e]).fetchall()
    counts = {name: 0 for name in ('cold_user', 'omitted_user', 'cold_movie',
                                   'omitted_movie', 'seen', 'eligible')}
    labels = []
    for user, movie, category in rows:
        counts[category] += 1
        if category == 'eligible':
            labels.append((user, movie))
    users = tuple(sorted({user for user, _ in labels}))
    pairs = tuple(labels)
    exclusions = tuple(counts.items())
    model_users = {raw for raw, _ in history.user_ids}
    positive_model_users = tuple(sorted({user for user, _, _ in rows if user in model_users}))
    seen = history.mask.seen_for(positive_model_users)
    no_candidates = sum(len(seen[user]) >= len(history.movie_ids) for user in positive_model_users)
    digest = stable_hash([history.digest, split, full_window, e, users, pairs, exclusions,
                          rating_events, positive_events, positive_users, no_candidates])
    return QualityBundle(history, split, full_window, e, users, pairs,
                         stable_hash(users), stable_hash(pairs),
                         exclusions, rating_events, positive_events, positive_users,
                         len(rows), positive_events - len(rows),
                         len(pairs) / len(rows) if rows else None,
                         len(users) / positive_users if positive_users else None,
                         no_candidates, digest)
