"""Synthetic CPU-only D2 demonstration; not the official 33-boundary runner."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import duckdb
import psutil

from src.artifacts import file_sha256
from src.benchmarking.common import RssSampler
from src.evaluation.refresh_boundary import (RankRequest, freeze_boundary,
                                             make_quality_bundle, select_serving_users)
from src.evaluation.refresh_results import (aggregate_split, evaluate_quality,
                                            paired_comparisons)
from src.evaluation.refresh_strategies import (ForwardLedger, VerifiedC0,
                                               create_p0, prepare, rank_users)
from src.graph.model_replay import ModelGraphReplay


ORDERS = {1: ('S0', 'S1', 'S2'), 2: ('S1', 'S2', 'S0'), 3: ('S2', 'S0', 'S1')}


def _month_start(epoch: int) -> bool:
    dt = datetime.fromtimestamp(epoch, timezone.utc)
    return (dt.day, dt.hour, dt.minute, dt.second) == (1, 0, 0, 0)


def _next_month(epoch: int) -> int:
    dt = datetime.fromtimestamp(epoch, timezone.utc)
    return int(datetime(dt.year + (dt.month == 12), dt.month % 12 + 1, 1,
                        tzinfo=timezone.utc).timestamp())


def run_toy(cohort: Path, events: Path, checkpoint: VerifiedC0, g0_snapshot: Path,
            windows: tuple[tuple[int, int], ...], output: Path) -> dict:
    """Exactly three tiny repetitions; all artifacts must be in a fresh toy dir."""
    if (not checkpoint.toy_cpu or checkpoint.provenance['node_count'] > 1000 or
            not 1 <= len(windows) <= 3 or
            any(not _month_start(b) or e != _next_month(b) or e > 1672531200
                for b, e in windows) or
            any(windows[i][1] != windows[i + 1][0] for i in range(len(windows) - 1))):
        raise ValueError('D2 coordinator accepts only small pre-holdout monthly CPU toys')
    with duckdb.connect() as con:
        event_count, last_event = con.execute('SELECT count(*),max(event_ts) FROM read_parquet(?)',
                                              [str(events)]).fetchone()
        if event_count > 10000 or last_event >= 1672531200:
            raise ValueError('D2 source is not a toy')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    p0_ledger = ForwardLedger()
    s1_ledger = ForwardLedger()
    p0_timings = {}
    report = dict(status='INCOMPLETE', repetitions=[], p0_setup_forward_attempts=0,
                  s1_forward_attempts=0, optimizer_updates=0, error=None)
    rep = b = e = strategy = None
    stage = 'p0_setup'
    try:
        p0 = output / 'p0'
        p0_identity, p0_manifest_sha = create_p0(
            checkpoint, g0_snapshot,
            base_boundary=checkpoint.provenance['base_cutoff_ts'],
            mapping_sha256=checkpoint.provenance['mapping_sha256'],
            source_sha256=checkpoint.provenance['source_sha256'],
            cache_path=p0, ledger=p0_ledger, timings=p0_timings)
        report['p0_setup_forward_attempts'] = p0_ledger.attempts
        report['p0_manifest_sha256'] = p0_manifest_sha
        report['p0_setup_s'] = perf_counter() - start
        report['p0_timings'] = p0_timings
        for rep in (1, 2, 3):
            rep_start = perf_counter()
            rep_rows = []
            quality_windows, quality_users = [], []
            rep_ledger_start = s1_ledger.attempts
            strategy = b = e = None
            stage = 'replay_initialization'
            with RssSampler(interval=.01) as rss:
                init_tick = perf_counter()
                with ModelGraphReplay.create(output / f'rep-{rep}.duckdb', cohort, events) as replay:
                    replay_initialization_s = perf_counter() - init_tick
                    for index, (b, e) in enumerate(windows):
                        stage = 'boundary_replay'
                        timing = dict(replay_s=0.0, snapshot_preparation_s=0.0,
                                      input_integrity_s=0.0, checkpoint_verify_s=0.0,
                                      snapshot_reload_materialize_s=0.0,
                                      cache_io_s=0.0, propagation_s=0.0,
                                      offline_rank_s=0.0, quality_bundle_s=0.0,
                                      quality_aggregation_s=0.0,
                                      serving_like_rank_s=0.0)
                        tick = perf_counter()
                        replay.advance(b)
                        timing['replay_s'] = perf_counter() - tick
                        tick = perf_counter()
                        history = freeze_boundary(replay, b, output / f'rep-{rep}-snapshot-{index}')
                        benchmark = select_serving_users(replay, history)
                        timing['snapshot_preparation_s'] = perf_counter() - tick
                        states, serving_rankings = {}, {}
                        strategy_timings = {}
                        for strategy in ORDERS[rep]:
                            stage = 'strategy_prepare'
                            scoped = {}
                            state = prepare(strategy, history, checkpoint, repetition=rep,
                                            cache_path=(output / f'rep-{rep}-s1-{index}')
                                            if strategy == 'S1' else None,
                                            p0_path=p0, p0_identity=p0_identity,
                                            p0_manifest_sha256=p0_manifest_sha,
                                            ledger=s1_ledger, timings=scoped)
                            states[strategy] = state
                            stage = 'serving_like_ranking'
                            tick = perf_counter()
                            serving_rankings[strategy] = rank_users(
                                state, RankRequest(history, benchmark.selected, 'serving_like'))
                            scoped['serving_like_rank_s'] = perf_counter() - tick
                            for key, seconds in scoped.items():
                                timing[key] = timing.get(key, 0.0) + seconds
                            strategy_timings[strategy] = scoped
                        strategy = None
                        row = dict(rep=rep, b=b, e=e, strategy_order=ORDERS[rep],
                                   history_sha256=history.digest,
                                   catalog_sha256=history.catalog_sha256,
                                   mask_sha256=history.mask.pairs_sha256,
                                   benchmark_pool=benchmark.eligible_pool_count,
                                   benchmark_selected=benchmark.selected,
                                   benchmark_selected_sha256=benchmark.selected_sha256,
                                   benchmark_status=benchmark.status,
                                   serving_rankings=serving_rankings,
                                   s1_forward_attempts=states['S1'].forward_attempts,
                                   optimizer_updates=0, timings=timing,
                                   strategy_timings=strategy_timings,
                                   cache_disposition={'S0': 'HIT', 'S1': 'MISS_PUBLISHED',
                                                      'S2': 'NOT_APPLICABLE'})
                        if rep == 1:
                            stage = 'quality_bundle'
                            tick = perf_counter()
                            bundle = make_quality_bundle(replay, history, e)
                            timing['quality_bundle_s'] = perf_counter() - tick
                            offline_rankings = {}
                            for strategy in ORDERS[rep]:
                                stage = 'offline_ranking'
                                tick = perf_counter()
                                offline_rankings[strategy] = rank_users(
                                    states[strategy], RankRequest(history, bundle.eligible_user_ids,
                                                                  'offline_quality'))
                                seconds = perf_counter() - tick
                                timing['offline_rank_s'] += seconds
                                strategy_timings[strategy]['offline_rank_s'] = seconds
                            strategy = None
                            stage = 'quality_aggregation'
                            tick = perf_counter()
                            quality = evaluate_quality(bundle, offline_rankings)
                            timing['quality_aggregation_s'] = perf_counter() - tick
                            quality_windows.extend(quality['windows'])
                            quality_users.extend(quality['per_user'])
                            row['quality_bundle_sha256'] = bundle.digest
                            row['coverage'] = quality['coverage']
                        rep_rows.append(row)
                        stage = 'boundary_complete'
                rep_record = dict(rep=rep, strategy_order=ORDERS[rep], windows=rep_rows,
                                  quality_windows=quality_windows, quality_users=quality_users,
                                  replay_initialization_s=replay_initialization_s,
                                  forward_attempts=s1_ledger.attempts - rep_ledger_start,
                                  optimizer_updates=0, process_peak_rss_bytes=rss.peak,
                                  host_available_ram_bytes=psutil.virtual_memory().available,
                                  status='COMPLETE')
            if rep == 1:
                stage = 'split_aggregation'
                tick = perf_counter()
                rep_record['split_macro'] = aggregate_split(quality_windows, split='toy')
                rep_record['paired'] = paired_comparisons(quality_users)
                rep_record['split_aggregation_s'] = perf_counter() - tick
            rep_record['elapsed_s'] = perf_counter() - rep_start
            report['repetitions'].append(rep_record)
        report['s1_forward_attempts'] = s1_ledger.attempts
        if (report['p0_setup_forward_attempts'] != 1 or
                report['s1_forward_attempts'] != 3 * len(windows) or
                p0_ledger.optimizer_updates != 0 or s1_ledger.optimizer_updates != 0):
            raise AssertionError('Toy forward budget mismatch')
        report['status'] = 'COMPLETE_TOY_CPU'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['s1_forward_attempts'] = s1_ledger.attempts
        report['failure'] = dict(rep=rep, b=b, e=e, strategy=strategy, stage=stage)
        if rep is not None and len(report['repetitions']) < rep:
            report['repetitions'].append(dict(rep=rep, status='INCOMPLETE',
                                              windows=rep_rows,
                                              forward_attempts=s1_ledger.attempts - rep_ledger_start,
                                              elapsed_s=perf_counter() - rep_start))
    finally:
        report['p0_setup_forward_attempts'] = p0_ledger.attempts
        report['optimizer_updates'] = p0_ledger.optimizer_updates + s1_ledger.optimizer_updates
        report_path = output / 'report.json'
        tick = perf_counter()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        report['output_write_s'] = perf_counter() - tick
        report['total_elapsed_s'] = perf_counter() - start
        # shortcut: final stamp excludes its own rewrite, use an external timer for real cost studies.
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report
