"""Thin CPU-testable S0/S1/S2 adapters over the existing rankers."""
from __future__ import annotations

import platform
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from src.artifacts import file_sha256
from src.evaluation.baselines import popularity_counts, popularity_topk
from src.evaluation.c0_validation import score_batched
from src.evaluation.refresh_boundary import BoundaryHistory, RankRequest, validate_history
from src.graph.model_replay import load_projection_snapshot
from src.models.c0 import C0Config, load_candidate
from src.models.lightgcn import to_sparse_tensor
from src.models.refresh_cache import load_verified, parameter_sha256, publish_atomic


@dataclass(frozen=True)
class VerifiedC0:
    path: Path
    sha256: str
    provenance: dict
    g0_manifest_sha256: str
    config: C0Config = C0Config()
    toy_cpu: bool = False

    def load(self):
        return load_candidate(self.path, expected_sha256=self.sha256,
                              config=self.config, provenance=self.provenance,
                              device='cpu', toy_cpu=self.toy_cpu)


@dataclass
class ForwardLedger:
    attempts: int = 0
    optimizer_updates: int = 0


@dataclass(frozen=True)
class StrategyState:
    strategy: str
    history: BoundaryHistory
    scores: object
    identity: dict | None
    cache_path: Path | None
    cache_manifest_sha256: str | None
    forward_attempts: int
    optimizer_updates: int = 0


@contextmanager
def _phase(timings: dict | None, key: str):
    start = perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + perf_counter() - start


def _identity(strategy: str, checkpoint: VerifiedC0, manifest: dict,
              snapshot_sha: str, repetition: int | None) -> dict:
    return dict(strategy=strategy, checkpoint_sha256=checkpoint.sha256,
                snapshot_sha256=snapshot_sha,
                mapping_sha256=manifest['mapping_sha256'],
                source_sha256=manifest['source_sha256'],
                cohort_sha256=manifest['cohort_manifest_sha256'],
                projection_version=manifest['projection_version'],
                boundary=manifest['observed_before_ts'],
                edges_sha256=manifest['edges_sha256'],
                degrees_sha256=manifest['degrees_sha256'],
                coo_sha256=manifest['coo_sha256'],
                node_count=manifest['shape'][0], dim=checkpoint.config.dim,
                layers=checkpoint.config.layers, dtype='torch.float32', device='cpu',
                environment=f'{platform.system()}:{platform.machine()}:torch-{torch.__version__}',
                repetition=repetition)


def _verified_snapshot(path: Path, *, mapping: str, source: str,
                       version: int, boundary: int):
    manifest, state, adjacency = load_projection_snapshot(
        path, expected_mapping=mapping, expected_version=version,
        expected_boundary=boundary, expected_source=source)
    return manifest, state, adjacency, file_sha256(path / 'manifest.json')


def create_p0(checkpoint: VerifiedC0, g0_snapshot: Path, *, base_boundary: int,
              mapping_sha256: str, source_sha256: str, cache_path: Path,
              ledger: ForwardLedger, timings: dict | None = None) -> tuple[dict, str]:
    """One explicitly budgeted setup forward; only synthetic CPU inputs in D2."""
    if cache_path.exists():
        raise FileExistsError('P0 already published; use verified load')
    with _phase(timings, 'snapshot_reload_materialize_s'):
        manifest, _, adjacency, snapshot_sha = _verified_snapshot(
            g0_snapshot, mapping=mapping_sha256, source=source_sha256,
            version=0, boundary=base_boundary)
    if (snapshot_sha != checkpoint.g0_manifest_sha256 or
            manifest['coo_sha256'] != checkpoint.provenance['snapshot_coo_sha256'] or
            manifest['edges_sha256'] != checkpoint.provenance['base_edges_sha256'] or
            manifest['edge_count'] != checkpoint.provenance['base_edge_count'] or
            manifest['cohort_manifest_sha256'] != checkpoint.provenance['cohort_manifest_sha256'] or
            base_boundary != checkpoint.provenance['base_cutoff_ts']):
        raise ValueError('C0 was not trained on this verified G0')
    with _phase(timings, 'checkpoint_verify_s'):
        model = checkpoint.load()
        before = parameter_sha256(model)
    identity = _identity('S0', checkpoint, manifest, snapshot_sha, None)
    ledger.attempts += 1
    with _phase(timings, 'propagation_s'):
        with torch.inference_mode():
            embeddings = model(to_sparse_tensor(adjacency, torch.device('cpu')))
    after = parameter_sha256(model)
    with _phase(timings, 'cache_io_s'):
        publish_atomic(cache_path, embeddings, identity, before, after)
    return identity, file_sha256(cache_path / 'manifest.json')


def prepare(strategy: str, history: BoundaryHistory, checkpoint: VerifiedC0,
            *, repetition: int, cache_path: Path | None = None,
            p0_path: Path | None = None, p0_identity: dict | None = None,
            p0_manifest_sha256: str | None = None,
            ledger: ForwardLedger | None = None,
            timings: dict | None = None) -> StrategyState:
    if strategy not in ('S0', 'S1', 'S2') or repetition not in (1, 2, 3):
        raise ValueError('Unknown strategy/repetition')
    with _phase(timings, 'input_integrity_s'):
        validate_history(history)
    if (history.mapping_sha256 != checkpoint.provenance['mapping_sha256'] or
            history.source_sha256 != checkpoint.provenance['source_sha256'] or
            history.cohort_sha256 != checkpoint.provenance['cohort_manifest_sha256']):
        raise ValueError('Checkpoint and projection provenance mismatch')
    if strategy == 'S0':
        if p0_path is None or p0_identity is None or p0_manifest_sha256 is None:
            raise ValueError('S0 needs the verified P0 identity')
        if (p0_identity['strategy'] != 'S0' or p0_identity['repetition'] is not None or
                p0_identity['checkpoint_sha256'] != checkpoint.sha256 or
                p0_identity['snapshot_sha256'] != checkpoint.g0_manifest_sha256 or
                p0_identity['coo_sha256'] != checkpoint.provenance['snapshot_coo_sha256'] or
                p0_identity['edges_sha256'] != checkpoint.provenance['base_edges_sha256'] or
                p0_identity['projection_version'] != 0 or
                p0_identity['boundary'] != checkpoint.provenance['base_cutoff_ts'] or
                p0_identity['dim'] != checkpoint.config.dim or
                p0_identity['layers'] != checkpoint.config.layers or
                p0_identity['dtype'] != 'torch.float32' or
                p0_identity['device'] != 'cpu' or
                p0_identity['environment'] != f'{platform.system()}:{platform.machine()}:torch-{torch.__version__}' or
                p0_identity['mapping_sha256'] != history.mapping_sha256 or
                p0_identity['source_sha256'] != history.source_sha256 or
                p0_identity['cohort_sha256'] != history.cohort_sha256 or
                p0_identity['node_count'] != len(history.user_ids) + len(history.movie_ids)):
            raise ValueError('P0 is incompatible with boundary history')
        with _phase(timings, 'checkpoint_verify_s'):
            model_hash = parameter_sha256(checkpoint.load())
        with _phase(timings, 'cache_io_s'):
            embeddings, cache_manifest = load_verified(
                p0_path, p0_identity, expected_manifest_sha256=p0_manifest_sha256)
        if cache_manifest['parameter_before'] != model_hash:
            raise ValueError('P0 parameter hash differs from C0')
        return StrategyState('S0', history, embeddings, p0_identity, p0_path,
                             p0_manifest_sha256, 0)
    with _phase(timings, 'snapshot_reload_materialize_s'):
        manifest, state, adjacency, snapshot_sha = _verified_snapshot(
            history.snapshot, mapping=history.mapping_sha256,
            source=history.source_sha256, version=history.projection_version,
            boundary=history.b)
    if (snapshot_sha != history.snapshot_sha256 or
            manifest['edges_sha256'] != history.edges_sha256 or
            manifest['degrees_sha256'] != history.degrees_sha256 or
            manifest['coo_sha256'] != history.coo_sha256 or
            manifest['cohort_manifest_sha256'] != history.cohort_sha256 or
            len(state['registry']) != len(history.user_ids) + len(history.movie_ids)):
        raise ValueError('Snapshot graph identity mismatch')
    if strategy == 'S2':
        with _phase(timings, 'popularity_counts_s'):
            counts = popularity_counts(state, dict(history.movie_ids))
        return StrategyState('S2', history, (state, counts), None, None, None, 0)
    if cache_path is None or ledger is None:
        raise ValueError('S1 needs an output cache and forward ledger')
    if cache_path.exists():
        raise FileExistsError('Scheduled refresh cannot reuse an earlier cache')
    with _phase(timings, 'checkpoint_verify_s'):
        model = checkpoint.load()
        before = parameter_sha256(model)
    identity = _identity('S1', checkpoint, manifest, snapshot_sha, repetition)
    ledger.attempts += 1
    with _phase(timings, 'propagation_s'):
        with torch.inference_mode():
            embeddings = model(to_sparse_tensor(adjacency, torch.device('cpu')))
    after = parameter_sha256(model)
    with _phase(timings, 'cache_io_s'):
        publish_atomic(cache_path, embeddings, identity, before, after)
        cache_sha = file_sha256(cache_path / 'manifest.json')
        verified, _ = load_verified(cache_path, identity, expected_manifest_sha256=cache_sha)
    return StrategyState('S1', history, verified, identity, cache_path, cache_sha, 1)


def get_scores(state: StrategyState):
    return dict(state.scores[1]) if state.strategy == 'S2' else state.scores.clone()


def rank_users(state: StrategyState, request: RankRequest):
    if request.history != state.history:
        raise ValueError('Ranking bundle differs from prepared state')
    users = dict(state.history.user_ids)
    movies = dict(state.history.movie_ids)
    seen = state.history.mask.seen_for(request.raw_user_ids)
    selected = {raw: users[raw] for raw in request.raw_user_ids}
    if state.strategy == 'S2':
        graph, counts = state.scores
        return popularity_topk(graph, selected, movies, seen, k=20, counts=counts)
    return score_batched(state.scores, selected, movies, seen, k=20)


def provenance(state: StrategyState) -> dict:
    return dict(strategy=state.strategy, history_sha256=state.history.digest,
                mapping_sha256=state.history.mapping_sha256,
                catalog_sha256=state.history.catalog_sha256,
                mask_sha256=state.history.mask.pairs_sha256,
                cache_path=str(state.cache_path) if state.cache_path else None,
                cache_manifest_sha256=state.cache_manifest_sha256,
                identity=state.identity)


def counters(state: StrategyState) -> dict[str, int]:
    return dict(propagation_forwards=state.forward_attempts,
                optimizer_updates=state.optimizer_updates)
