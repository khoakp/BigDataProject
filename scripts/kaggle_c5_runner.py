"""C5 validation only: one G0 forward per candidate, no C0 selection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import duckdb
import psutil
import torch

from scripts.kaggle_c3_runner import available_ram, disk_bytes, verify_inputs
from scripts.kaggle_c4_runner import locate_c3_snapshot
from scripts.verify_kaggle_c4 import verify as verify_c4
from src.artifacts import file_sha256
from src.evaluation.c0_validation import (START, END, aggregate_validation,
                                           boundaries, evaluate_embeddings,
                                           score_batched, validation_windows)
from src.graph.equivalence import checksum
from src.models.c0 import C0Config, STEPS, load_candidate, load_g0


ROOT = Path('/kaggle/working/goal04_c5')
INPUT_ROOT = Path('/kaggle/input')
SOURCE_ROOT = Path(__file__).resolve().parents[1]
MAX_PHASE_C_S = 4 * 3600
FINALIZATION_MARGIN_S = 120
C4_REPORT_SHA = '3d9addd49c432432f036275b4c243c78041c929e53e386fbee376e12c3cae5ce'
C4_PACKAGE = SOURCE_ROOT/'results/goal04_c4/run_04'
MONTHLY_CSV = SOURCE_ROOT/'results/goal04_protocol/monthly.csv'


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def locate_c4() -> Path:
    matches = []
    for candidate in INPUT_ROOT.rglob('candidate_0.pt'):
        root = candidate.parent.parent
        report = root/'report.json'
        if report.is_file() and file_sha256(report) == C4_REPORT_SHA:
            matches.append(root)
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one approved C4 candidate set; found {len(matches)}')
    return matches[0]


def load_2020_windows(canonical_path: Path, registry, edges, monthly_csv: Path):
    """Read 2020 only; compare fixed eligible counts with Phase A profile."""
    with duckdb.connect(config={'memory_limit':'2GB','threads':'2'}) as con:
        events = con.execute('''SELECT event_id,event_ts,user_raw_id,movie_raw_id,
            rating,source_row_id FROM read_parquet(?)
            WHERE event_ts>=? AND event_ts<? ORDER BY event_id''',
            [str(canonical_path), START, END]).fetchall()
    windows = validation_windows(events, registry, edges)
    with monthly_csv.open(encoding='utf-8') as stream:
        expected = {row['window']: row for row in csv.DictReader(stream)
                    if row['window'].startswith('2020-')}
    if (len(windows) != 12 or len(expected) != 12 or
        [w['start'] for w in windows] != boundaries()[:-1]):
        raise ValueError('Incomplete 2020 validation calendar')
    for index, window in enumerate(windows, 1):
        profile = expected[f'2020-{index:02d}']
        if (window['counts']['eligible'] != int(profile['eligible_pairs']) or
            len(window['relevant']) != int(profile['eligible_users'])):
            raise ValueError(f'2020-{index:02d} eligible universe differs from approved profile')
    return windows


def device_equivalence(emb0, emb1, registry, windows, *, atol=1e-5):
    """Check identical Top-20 order and numerical scores on both T4 devices."""
    users = {raw: node for node, kind, raw, *_ in registry if kind == 'user'}
    movies = {raw: node for node, kind, raw, *_ in registry if kind == 'movie'}
    sample = sorted(windows[0]['relevant'])[:16]
    if not sample:
        raise ValueError('No eligible numerical gate users')
    selected = {raw: users[raw] for raw in sample}
    left = score_batched(emb0, selected, movies, windows[0]['seen'], batch_size=16)
    right = score_batched(emb1, selected, movies, windows[0]['seen'], batch_size=16)
    maximum = 0.0
    for user in sample:
        if [m for m, _ in left[user]] != [m for m, _ in right[user]]:
            return False, None
        for (_, a), (_, b) in zip(left[user], right[user]):
            maximum = max(maximum, abs(a-b))
    return maximum <= atol, maximum


def _synchronize(emb1) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
        if emb1 is not None and emb1.device.type == 'cuda':
            torch.cuda.synchronize(emb1.device)


def evaluate_candidate(emb0, emb1, registry, windows):
    """Partition months 1–6/7–12 after the gate; fallback without a new forward."""
    _synchronize(emb1)
    started = time.perf_counter()
    if emb1 is not None:
        equivalent, maximum = device_equivalence(emb0, emb1, registry, windows)
        gate_status = 'PASS' if equivalent else 'FAIL_FALLBACK'
    else:
        equivalent, maximum = False, None
        gate_status = 'ONE_GPU'
    _synchronize(emb1)
    gate_s = time.perf_counter()-started
    started = time.perf_counter()
    if equivalent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(evaluate_embeddings, emb0, registry, windows,
                               selected=tuple(range(6)), batch_size=32)
            right = pool.submit(evaluate_embeddings, emb1, registry, windows,
                                selected=tuple(range(6,12)), batch_size=32)
            rows = [left.result(), right.result()]
        mode = 'dual_t4'
    else:
        rows = [evaluate_embeddings(emb0, registry, windows, batch_size=32)]
        mode = 'single_t4_fallback'
    _synchronize(emb1)
    ranking_s = time.perf_counter()-started
    return (aggregate_validation(rows), mode,
            {'status':gate_status,'max_abs_top20_score_diff':maximum,
             'sampled_users':min(16,len(windows[0]['relevant'])) if emb1 is not None else 0,
             'atol':1e-5},
            {'numerical_gate_s':gate_s,'ranking_and_metrics_s':ranking_s})


def _matrix_row(step, metric, mode, gate, provenance, catalog_hash):
    return {'step':step,'window_index':metric['index'],
            'start_utc':datetime.fromtimestamp(metric['start'],timezone.utc).isoformat(),
            'end_utc':datetime.fromtimestamp(metric['end'],timezone.utc).isoformat(),
            'start_ts':metric['start'],'end_exclusive_ts':metric['end'],
            'eligible_users':metric['users'],'eligible_labels':metric['labels'],
            'recall_20':metric['recall_20'],'ndcg_20':metric['ndcg_20'],
            'catalog_movies':40_666,'catalog_mapping_sha256':catalog_hash,
            'mask_policy':'rolling_observed_positive_before_boundary',
            'mask_boundary_ts':metric['start'],
            'mask_base_edges_sha256':provenance['base_edges_sha256'],
            'mask_canonical_source_sha256':provenance['source_sha256'],
            'gpu_mode':mode,'numerical_equivalence':gate['status']}


def _gpu_metrics():
    return [{'device':f'cuda:{device}',
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
             'free_bytes':torch.cuda.mem_get_info(device)[0],
             'total_bytes':torch.cuda.mem_get_info(device)[1]}
            for device in range(torch.cuda.device_count())]


def worker() -> None:
    started = time.monotonic()
    report_path = ROOT/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    prior = report['prior_verified_kaggle_elapsed_s']
    report['stage'] = 'input_verification'
    _write(report_path,report)

    def budget_check():
        if prior+time.monotonic()-report['job_started_monotonic'] >= MAX_PHASE_C_S-FINALIZATION_MARGIN_S:
            raise TimeoutError('Phase C budget safety margin reached')

    try:
        if torch.cuda.get_device_name(0) != 'Tesla T4':
            raise RuntimeError('Approved cuda:0 Tesla T4 unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)
        snapshot,_ = locate_c3_snapshot()
        inputs = verify_inputs()
        c4_root = locate_c4()
        verification = verify_c4(c4_root,C4_PACKAGE)
        c4 = json.loads((c4_root/'candidates/report.json').read_text(encoding='utf-8'))
        if [row['step'] for row in c4['checkpoints']] != list(STEPS):
            raise ValueError('C4 candidates are incomplete')
        provenance = c4['provenance']
        report['phase_timings_s']['input_verification'] = time.monotonic()-started
        budget_check()

        report['stage'] = 'g0_load'
        _write(report_path,report)
        phase = time.monotonic()
        manifest,state,sparse = load_g0(snapshot,
            source_sha256=provenance['source_sha256'],
            mapping_sha256=provenance['mapping_sha256'],device='cuda:0')
        catalog_hash = provenance['mapping_sha256']
        if (file_sha256(snapshot/'manifest.json') != provenance['snapshot_manifest_sha256'] or
            inputs['canonical_sha256'] != provenance['source_sha256'] or
            inputs['cohort_manifest_sha256'] != provenance['cohort_manifest_sha256'] or
            manifest['coo_sha256'] != provenance['snapshot_coo_sha256'] or
            checksum([row[:3] for row in state['registry']]) != catalog_hash or
            len(state['registry']) != 90_253 or len(state['edges']) != 4_999_981 or
            sum(row[1] == 'movie' for row in state['registry']) != 40_666):
            raise ValueError('C5 source, graph, mapping or catalog mismatch')
        report['phase_timings_s']['g0_load'] = time.monotonic()-phase
        budget_check()

        report['stage'] = 'window_build'
        _write(report_path,report)
        phase = time.monotonic()
        windows = load_2020_windows(Path(inputs['canonical_path']),
                                    state['registry'],state['edges'],MONTHLY_CSV)
        report['phase_timings_s']['window_build'] = time.monotonic()-phase
        report['inputs'] = {'c4_verification_status':verification['status'],
                            'c4_report_sha256':C4_REPORT_SHA,
                            'snapshot_manifest_sha256':provenance['snapshot_manifest_sha256'],
                            'mapping_sha256':catalog_hash,
                            'base_edges_sha256':provenance['base_edges_sha256'],
                            'canonical_sha256':inputs['canonical_sha256'],
                            'monthly_csv_sha256':file_sha256(MONTHLY_CSV),
                            'windows':len(windows),'catalog_movies':40_666}
        _write(report_path,report)
        budget_check()

        for candidate in c4['checkpoints']:
            step = candidate['step']
            report['stage'] = f'checkpoint_{step}'
            _write(report_path,report)
            timings = {}
            phase = time.monotonic()
            model = load_candidate(c4_root/'candidates'/candidate['file'],
                expected_sha256=candidate['sha256'],config=C0Config(),
                provenance=provenance,device='cuda:0')
            timings['checkpoint_load_s'] = time.monotonic()-phase
            budget_check()
            torch.cuda.synchronize(0)
            phase = time.perf_counter()
            report['propagation_forwards'] += 1
            _write(report_path,report)
            with torch.inference_mode():
                emb0 = model(sparse).detach().clone()
            torch.cuda.synchronize(0)
            timings['propagation_s'] = time.perf_counter()-phase
            del model
            if not bool(torch.isfinite(emb0).all()):
                raise ValueError('Nonfinite propagated vectors')
            budget_check()
            phase = time.perf_counter()
            emb1 = emb0.to('cuda:1',copy=True) if (
                torch.cuda.device_count() > 1 and
                torch.cuda.get_device_name(1) == 'Tesla T4') else None
            _synchronize(emb1)
            timings['vector_copy_s'] = time.perf_counter()-phase
            budget_check()
            scores,mode,gate,evaluation_timings = evaluate_candidate(
                emb0,emb1,state['registry'],windows)
            timings.update(evaluation_timings)
            budget_check()
            phase = time.monotonic()
            vector_sha = hashlib.sha256(emb0.cpu().numpy().tobytes()).hexdigest()
            rows = [_matrix_row(step,row,mode,gate,provenance,catalog_hash)
                    for row in scores['windows']]
            if len(rows) != 12:
                raise ValueError('Incomplete monthly candidate evaluation')
            report['validation_matrix'].extend(rows)
            report['candidate_scores'].append({
                'step':step,'checkpoint_sha256':candidate['sha256'],
                'propagated_vectors_sha256':vector_sha,
                'validation_recall_20':scores['validation_recall_20'],
                'validation_ndcg_20':scores['validation_ndcg_20'],
                'window_count':len(rows),'gpu_mode':mode,
                'numerical_equivalence':gate,'timings_s':timings})
            report['gpu_resources'] = _gpu_metrics()
            if any(row['free_bytes']/row['total_bytes'] < .30
                   for row in report['gpu_resources'][:2]):
                raise MemoryError('GPU effective headroom below 30%')
            _write(ROOT/'validation-matrix.json',report['validation_matrix'])
            report['phase_timings_s'][f'checkpoint_{step}_artifact_write'] = time.monotonic()-phase
            _write(report_path,report)
            budget_check()
            del emb0,emb1

        if ([row['step'] for row in report['candidate_scores']] != list(STEPS) or
            len(report['validation_matrix']) != 60 or
            report['propagation_forwards'] != 5 or report['optimizer_updates'] != 0):
            raise ValueError('Incomplete validation matrix or forward budget mismatch')
        with (ROOT/'candidate-comparison.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,['step','validation_recall_20',
                'validation_ndcg_20','window_count','gpu_mode','checkpoint_sha256'])
            writer.writeheader()
            for row in report['candidate_scores']:
                writer.writerow({key:row[key] for key in writer.fieldnames})
        report['artifact_sha256'] = {
            'validation-matrix.json':file_sha256(ROOT/'validation-matrix.json'),
            'candidate-comparison.csv':file_sha256(ROOT/'candidate-comparison.csv')}
        report['stage'] = 'complete'
        report['status'] = 'COMPLETE_VALIDATION'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['worker_elapsed_s'] = time.monotonic()-started
        report['cpu_rss_end_bytes'] = psutil.Process().memory_info().rss
        if torch.cuda.is_available():
            report['gpu_resources'] = _gpu_metrics()
        _write(report_path,report)


def run() -> None:
    prior_text = os.environ.get('GOAL04_VERIFIED_PRIOR_PHASE_C_S')
    if prior_text is None:
        raise RuntimeError('C5 budget gate: verified prior Kaggle elapsed is missing')
    prior = float(prior_text)
    if not math.isfinite(prior) or not 0 <= prior < MAX_PHASE_C_S-FINALIZATION_MARGIN_S:
        raise ValueError('C5 budget gate: invalid verified prior elapsed')
    ROOT.mkdir(parents=True,exist_ok=False)
    started = float(os.environ.get('GOAL04_C5_JOB_STARTED_MONOTONIC',time.monotonic()))
    report = {'status':'INCOMPLETE','stage':'initialization',
              'job_started_monotonic':started,
              'prior_verified_kaggle_elapsed_s':prior,
              'budget_audit_sha256':os.environ['GOAL04_C5_BUDGET_AUDIT_SHA256'],
              'git_head':os.environ['GOAL04_C5_GIT_HEAD'],
              'source_bundle_sha256':os.environ['GOAL04_C5_SOURCE_SHA256'],
              'candidate_scores':[],'validation_matrix':[],
              'propagation_forwards':0,'optimizer_updates':0,
              'phase_timings_s':{},
              'environment':{'python':platform.python_version(),
                             'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                             'duckdb':duckdb.__version__,
                             'gpu_names':[torch.cuda.get_device_name(i)
                                          for i in range(torch.cuda.device_count())],
                             'cpu_count':os.cpu_count()}}
    _write(ROOT/'report.json',report)
    peak_rss = peak_disk = 0
    min_available = available_ram()
    min_disk_free = psutil.disk_usage(ROOT).free
    reason = None
    with (ROOT/'worker.log').open('w',encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable,'-m','scripts.kaggle_c5_runner','--worker'],
                                 cwd=SOURCE_ROOT,stdout=stream,stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                family = [process]+process.children(recursive=True)
                peak_rss = max(peak_rss,sum(p.memory_info().rss
                    for p in family if p.is_running()))
            except psutil.NoSuchProcess:
                family = [process]
            min_available = min(min_available,available_ram())
            min_disk_free = min(min_disk_free,psutil.disk_usage(ROOT).free)
            peak_disk = max(peak_disk,disk_bytes(ROOT))
            if prior+time.monotonic()-started >= MAX_PHASE_C_S-FINALIZATION_MARGIN_S:
                reason = 'PHASE_C_BUDGET_MARGIN'
            elif min_available < 2*1024**3:
                reason = 'CPU_AVAILABLE_BELOW_2_GIB'
            elif min_disk_free < 2*1024**3:
                reason = 'DISK_FREE_BELOW_2_GIB'
            if reason:
                for member in reversed(family):
                    try: member.terminate()
                    except psutil.NoSuchProcess: pass
                break
            time.sleep(.2)
        code = child.wait()
    report = json.loads((ROOT/'report.json').read_text(encoding='utf-8'))
    elapsed = time.monotonic()-started
    report.update(job_elapsed_s=elapsed,
                  cumulative_phase_c_kaggle_elapsed_s=prior+elapsed,
                  peak_process_tree_rss_bytes=peak_rss,
                  min_effective_available_ram_bytes=min_available,
                  peak_output_disk_bytes=peak_disk,
                  min_disk_free_bytes=min_disk_free,
                  worker_exit_code=code,stop_reason=reason)
    if (code or reason or report['status'] != 'COMPLETE_VALIDATION' or
        len(report['validation_matrix']) != 60 or
        report['propagation_forwards'] != 5 or report['optimizer_updates'] != 0 or
        report['cumulative_phase_c_kaggle_elapsed_s'] >= MAX_PHASE_C_S-FINALIZATION_MARGIN_S):
        report['status'] = 'INCOMPLETE'
    _write(ROOT/'report.json',report)
    print(json.dumps({'status':report['status'],
        'job_elapsed_s':report['job_elapsed_s'],
        'matrix_rows':len(report['validation_matrix']),
        'propagation_forwards':report['propagation_forwards']}),flush=True)
    if report['status'] != 'COMPLETE_VALIDATION':
        raise SystemExit(1)


if __name__ == '__main__':
    if '--worker' in sys.argv: worker()
    else: run()
