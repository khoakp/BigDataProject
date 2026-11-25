"""Approval-gated, single-boundary resource preflight; never evaluates labels."""
from __future__ import annotations

import json
import os
import signal
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.kaggle_c3_runner import available_ram, disk_bytes, verify_inputs
from src.artifacts import file_sha256
from src.benchmarking.common import PreflightSampler
from src.evaluation.refresh_boundary import RankRequest, freeze_boundary, select_serving_users
from src.evaluation.refresh_strategies import ForwardLedger, VerifiedC0, create_p0, prepare, rank_users
from src.graph.model_replay import ModelGraphReplay, load_projection_snapshot
from src.models.c0 import C0Config, load_candidate


INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working/goal04_d4')
BOUNDARY = 1609459200
C3_REPORT_SHA = '29cc7adeb929b3b53a8ffdbb151c73acf28ba84430ff5d4026dca18ab224f26b'
SELECTION = Path(__file__).resolve().parents[1] / 'results/goal04_training/selection.json'


def approval_gate(root: Path = INPUT, *, bundle_sha256: str | None = None,
                  git_head: str | None = None) -> dict:
    """A private Owner decision must name the exact package and two forward attempts."""
    matches = list(root.rglob('d4-owner-approval.json')) if root.is_dir() else []
    if len(matches) != 1:
        raise PermissionError('D4 GPU blocked: expected one private Owner approval')
    decision = json.loads(matches[0].read_text(encoding='utf-8'))
    if (not bundle_sha256 or not git_head or
            decision.get('approved_source_bundle_sha256') != bundle_sha256 or
            decision.get('approved_git_head') != git_head or
            not isinstance(decision.get('owner_decision_reference'), str) or
            not decision['owner_decision_reference'].strip() or
            decision.get('approved_boundary') != BOUNDARY or
            decision.get('approved_device') != 'cuda:0' or
            type(decision.get('approved_job_count')) is not int or
            decision['approved_job_count'] != 1 or
            type(decision.get('approved_forward_attempts')) is not int or
            decision['approved_forward_attempts'] != 2 or
            decision.get('p0_accounting') != 'official_setup_1_of_100' or
            decision.get('s1_accounting') != 'official_rep1_boundary1_1_of_99' or
            decision.get('s1_reuse_authorized') is not True or
            type(decision.get('max_wall_s')) is not int or
            not 0 < decision['max_wall_s'] <= 3600 or
            type(decision.get('min_host_available_bytes')) is not int or
            decision['min_host_available_bytes'] < 8 * 1024**3 or
            type(decision.get('min_disk_free_bytes')) is not int or
            decision['min_disk_free_bytes'] < 10 * 1024**3 or
            decision.get('min_gpu_free_fraction') != .30):
        raise PermissionError('D4 GPU blocked: approval metadata does not match exact proposal')
    return decision


def _unique_file(name: str, sha256: str) -> Path:
    matches = [path for path in INPUT.rglob(name) if file_sha256(path) == sha256]
    if len(matches) != 1:
        raise ValueError(f'Expected one approved {name}; found {len(matches)}')
    return matches[0]


def _write(report: dict) -> None:
    tick = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / 'report.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(target)
    timings = report.get('timings_s')
    if timings is not None:
        timings['output_write_s'] = timings.get('output_write_s', 0.0) + time.monotonic() - tick


def _limits(decision: dict, start: float, stage: str) -> None:
    if time.monotonic() - start > decision['max_wall_s']:
        raise TimeoutError(f'Total wall cap exceeded at {stage}')
    if available_ram() < decision['min_host_available_bytes']:
        raise MemoryError(f'Host RAM below approved minimum at {stage}')
    if shutil.disk_usage(OUTPUT.parent).free < decision['min_disk_free_bytes']:
        raise OSError(f'Disk capacity below approved minimum at {stage}')
    free, total = torch.cuda.mem_get_info(0)
    if free / total < decision['min_gpu_free_fraction']:
        raise MemoryError(f'GPU headroom below approved minimum at {stage}')


@contextmanager
def _deadline(decision: dict, start: float, phase_s: int):
    """Linux preflight interrupt for this process only; no child/process-group kill."""
    remaining = decision['max_wall_s'] - (time.monotonic() - start)
    if remaining <= 0:
        raise TimeoutError('D4 total wall cap exceeded')
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(*_):
        raise TimeoutError('D4 phase or total wall cap exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, phase_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _c0(selection: dict, checkpoint_path: Path, g0_path: Path) -> VerifiedC0:
    if file_sha256(checkpoint_path) != selection['checkpoint']['sha256']:
        raise ValueError('Original C0 SHA mismatch')
    if file_sha256(g0_path / 'manifest.json') != selection['g0_snapshot_manifest_sha256']:
        raise ValueError('G0 manifest SHA mismatch')
    g0 = json.loads((g0_path / 'manifest.json').read_text(encoding='utf-8'))
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    provenance = payload['provenance']
    expected = {'mapping_sha256': selection['mapping_sha256'],
                'cohort_manifest_sha256': selection['cohort_manifest_sha256'],
                'snapshot_manifest_sha256': selection['g0_snapshot_manifest_sha256'],
                'snapshot_coo_sha256': selection['snapshot_coo_sha256'],
                'source_sha256': selection['canonical_source_sha256'],
                'c3_report_sha256': C3_REPORT_SHA,
                'base_cutoff_ts': 1577836800, 'node_count': 90253,
                'base_edge_count': 4999981,
                'base_edges_sha256': selection['base_edges_sha256'],
                'base_degrees_sha256': g0['degrees_sha256'],
                'git_head': selection['training_git_head'],
                'source_bundle_sha256': selection['training_source_bundle_sha256']}
    if provenance != expected or payload['step'] != 1500:
        raise ValueError('C0 provenance mismatch')
    verified = VerifiedC0(checkpoint_path, selection['checkpoint']['sha256'],
                          provenance, selection['g0_snapshot_manifest_sha256'])
    load_candidate(checkpoint_path, expected_sha256=verified.sha256,
                   config=C0Config(), provenance=provenance, device='cpu')
    load_projection_snapshot(g0_path, expected_mapping=selection['mapping_sha256'],
                             expected_version=0, expected_boundary=1577836800,
                             expected_source=selection['canonical_source_sha256'],
                             materialize=False)
    return verified


def run() -> dict:
    start = time.monotonic()
    report = {'status': 'INCOMPLETE', 'stage': 'approval', 'error': None,
              'forward_attempts': 0, 'optimizer_updates': 0, 'timings_s': {},
              'cpu_s': {}, 'phase_resources': {},
              'gpu_executed': False, 'duckdb_memory': None,
              'duckdb_unavailable_reason': 'Replay has not completed',
              'input_bytes': None, 'resources': None}
    cpu_start = time.process_time()
    sampler = None
    owned_output = False
    ledger = ForwardLedger()
    try:
        decision = approval_gate(bundle_sha256=os.environ.get('GOAL04_D4_BUNDLE_SHA256'),
                                 git_head=os.environ.get('GOAL04_D4_GIT_HEAD'))
        OUTPUT.mkdir(parents=True, exist_ok=False)
        owned_output = True
        _write(report)
        if (not torch.cuda.is_available() or torch.cuda.device_count() < 1 or
                torch.cuda.current_device() != 0 or
                torch.cuda.get_device_name(0) != 'Tesla T4'):
            raise ValueError('Unsupported GPU: Tesla T4 at cuda:0 required')
        report['gpu_device'] = torch.cuda.get_device_name(0)
        _limits(decision, start, 'input_verification')
        spill = OUTPUT / 'spill'
        with PreflightSampler(interval=.1, device='cuda:0', spill_dir=spill) as sampler:
            selection = json.loads(SELECTION.read_text(encoding='utf-8'))
            if selection['status'] != 'PHASE_C_COMPLETE':
                raise ValueError('Selection status mismatch')
            tick = time.monotonic()
            cpu_tick = time.process_time()
            with _deadline(decision, start, 900):
                inputs = verify_inputs()
            if (inputs['canonical_sha256'] != selection['canonical_source_sha256'] or
                    file_sha256(Path(inputs['cohort_path']) / 'manifest.json') !=
                    selection['cohort_manifest_sha256']):
                raise ValueError('C3 input identity mismatch')
            g0_manifest = _unique_file('manifest.json', selection['g0_snapshot_manifest_sha256'])
            c0_file = _unique_file('candidate_1500.pt', selection['checkpoint']['sha256'])
            with _deadline(decision, start, 900):
                checkpoint = _c0(selection, c0_file, g0_manifest.parent)
            report['timings_s']['input_verification'] = time.monotonic() - tick
            report['cpu_s']['input_verification'] = time.process_time() - cpu_tick
            report['input_sha256'] = {'c0': checkpoint.sha256,
                'g0_manifest': checkpoint.g0_manifest_sha256,
                'canonical': inputs['canonical_sha256'],
                'cohort_manifest': selection['cohort_manifest_sha256']}
            report['input_bytes'] = (c0_file.stat().st_size +
                Path(inputs['canonical_path']).stat().st_size +
                disk_bytes(Path(inputs['cohort_path'])) + disk_bytes(g0_manifest.parent))
            _limits(decision, start, 'replay_initialization')
            report['stage'] = 'replay_initialization'
            _write(report)
            tick = time.monotonic()
            cpu_tick = time.process_time()
            spill.mkdir()
            graph_profile = {}
            with _deadline(decision, start, 300):
                graph = ModelGraphReplay.create(OUTPUT / 'replay.duckdb',
                                         Path(inputs['cohort_path']),
                                         Path(inputs['canonical_path']),
                                         duckdb_config={'memory_limit': '1GB',
                                                        'threads': '1',
                                                        'temp_directory': str(spill)},
                                         profile_timings=graph_profile)
            with graph as replay:
                report['timings_s']['replay_initialization'] = time.monotonic() - tick
                report['cpu_s']['replay_initialization'] = time.process_time() - cpu_tick
                _limits(decision, start, 'boundary_freeze')
                report['stage'] = 'boundary_freeze'
                _write(report)
                tick = time.monotonic()
                cpu_tick = time.process_time()
                with _deadline(decision, start, 300):
                    history = freeze_boundary(replay, BOUNDARY, OUTPUT / 'snapshot',
                                              profile_timings=graph_profile)
                report['timings_s']['graph_profile'] = graph_profile
                report['timings_s']['replay_snapshot_verify'] = time.monotonic() - tick
                report['cpu_s']['replay_snapshot_verify'] = time.process_time() - cpu_tick
                if report['timings_s']['replay_snapshot_verify'] > 300:
                    raise TimeoutError('Graph/snapshot setup exceeded 300 s')
                benchmark = select_serving_users(replay, history)
                if len(history.user_ids) != 49587 or len(history.movie_ids) != 40666:
                    raise ValueError('Fixed cohort/catalog mismatch')
                report['benchmark'] = {'eligible_pool': benchmark.eligible_pool_count,
                                       'selected_count': len(benchmark.selected),
                                       'selected_sha256': benchmark.selected_sha256}
                report['snapshot_manifest_sha256'] = history.snapshot_sha256
                report['stage'] = 'p0_setup'
                _limits(decision, start, report['stage'])
                _write(report)
                p0_timing = {}
                cpu_tick = time.process_time()
                with _deadline(decision, start, 300):
                    p0_identity, p0_sha = create_p0(checkpoint, g0_manifest.parent,
                        base_boundary=1577836800, mapping_sha256=selection['mapping_sha256'],
                        source_sha256=selection['canonical_source_sha256'],
                        cache_path=OUTPUT / 'p0', ledger=ledger, timings=p0_timing, device='cuda:0')
                report['forward_attempts'] = ledger.attempts
                report['timings_s']['p0'] = p0_timing
                report['p0_manifest_sha256'] = p0_sha
                if p0_timing.get('propagation_s', 0) > 60:
                    raise TimeoutError('P0 propagation exceeded proposed 60 s cap')
                report['cpu_s']['p0'] = time.process_time() - cpu_tick
                sampler.sample()
                report['phase_resources']['p0'] = sampler.record()
                report['stage'] = 'strategy_preparation'
                _limits(decision, start, report['stage'])
                _write(report)
                rankings = {}
                for strategy in ('S0', 'S1', 'S2'):
                    timing = {}
                    cpu_tick = time.process_time()
                    with _deadline(decision, start, 300):
                        state = prepare(strategy, history, checkpoint, repetition=1,
                            cache_path=OUTPUT / 's1' if strategy == 'S1' else None,
                            p0_path=OUTPUT / 'p0', p0_identity=p0_identity,
                            p0_manifest_sha256=p0_sha, ledger=ledger,
                            timings=timing, device='cuda:0')
                    report['forward_attempts'] = ledger.attempts
                    report['timings_s'][strategy] = timing
                    if strategy == 'S1':
                        report['s1_manifest_sha256'] = state.cache_manifest_sha256
                    report['cpu_s'][strategy] = time.process_time() - cpu_tick
                    sampler.sample()
                    report['phase_resources'][f'{strategy}_prepare'] = sampler.record()
                    if strategy == 'S1' and timing.get('propagation_s', 0) > 60:
                        raise TimeoutError('S1 propagation exceeded 60 s')
                    if strategy == 'S1' and (
                        report['timings_s']['replay_snapshot_verify'] +
                        timing.get('snapshot_reload_materialize_s', 0) +
                        timing.get('sparse_materialization_s', 0) +
                        timing.get('cpu_to_gpu_transfer_s', 0) > 300):
                        raise TimeoutError('Boundary graph/snapshot/transfer exceeded 300 s')
                    _limits(decision, start, f'{strategy}_ranking')
                    tick = time.monotonic()
                    cpu_tick = time.process_time()
                    with _deadline(decision, start, 300):
                        ranked = rank_users(state,
                            RankRequest(history, benchmark.selected, 'serving_like'),
                            timings=timing)
                    torch.cuda.synchronize(0)
                    report['timings_s'][strategy]['serving_like_rank_s'] = time.monotonic() - tick
                    report['cpu_s'][f'{strategy}_serving_like_rank'] = time.process_time() - cpu_tick
                    sampler.sample()
                    report['phase_resources'][f'{strategy}_rank'] = sampler.record()
                    if report['timings_s'][strategy]['serving_like_rank_s'] > 300:
                        raise TimeoutError('Serving ranking exceeded 300 s')
                    rankings[strategy] = {'users': len(ranked),
                        'top20_sha256': __import__('hashlib').sha256(
                            json.dumps(ranked, sort_keys=True).encode()).hexdigest()}
                    _write(report)
                report['rankings'] = rankings
                report['snapshot_bytes'] = disk_bytes(OUTPUT / 'snapshot')
                report['cache_bytes'] = disk_bytes(OUTPUT / 'p0') + disk_bytes(OUTPUT / 's1')
                report['duckdb_memory'] = replay.con.execute('SELECT * FROM duckdb_memory()').fetchall()
                report['duckdb_unavailable_reason'] = None
                report['duckdb_memory_limit'] = '1GB'
                report['duckdb_spill_directory'] = str(spill)
                if ledger.attempts != 2 or ledger.optimizer_updates != 0:
                    raise ValueError('Forward/optimizer budget mismatch')
            if sampler.gpu_error:
                raise RuntimeError(f'GPU telemetry unavailable: {sampler.gpu_error}')
            report['stage'] = 'complete'
            report['status'] = 'COMPLETE_PREFLIGHT'
            report['gpu_executed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['forward_attempts'] = ledger.attempts
        report['gpu_executed'] = ledger.attempts > 0
    finally:
        report['elapsed_s'] = time.monotonic() - start
        report['total_cpu_s'] = time.process_time() - cpu_start
        report['process_exit_code'] = 0 if report['status'] == 'COMPLETE_PREFLIGHT' else 1
        if sampler is not None:
            report['resources'] = sampler.record()
        if owned_output:
            report['output_bytes'] = disk_bytes(OUTPUT)
            _write(report)
    return report


if __name__ == '__main__':
    result = run()
    raise SystemExit(0 if result['status'] == 'COMPLETE_PREFLIGHT' else 1)
