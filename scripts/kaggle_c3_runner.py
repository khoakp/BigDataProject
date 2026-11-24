"""Kaggle C3 preflight: verified real-5M projection, no model or optimizer."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import psutil

from src.artifacts import file_sha256
from src.graph.model_replay import ModelGraphReplay, load_projection_snapshot


ROOT = Path('/kaggle/working/goal04_c3')
INPUT_ROOT = Path('/kaggle/input')
SOURCE_SHA = '91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee'
CANONICAL_SHA = '494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e'
MANIFEST_SHA = 'be1d5264abcc7ead16964d86238b6a9944c72ffe1a80493f1d57551c5528c4f9'
MAPPING_SHA = '996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9'
BASE_EDGES_SHA = 'fb2c7fe7d64e018bebcd7101df41249dbfa57e1db5369fae01d023e9c2e5392a'
BASE_DEGREES_SHA = 'ee423151ba7070bc616b8a43475322410b759297c086b9331ba4f3318f49e2b4'
BASE_BOUNDARY = 1577836800
MONTH_BOUNDARY = 1580515200
MAX_ELAPSED = 4 * 3600


def disk_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def cgroup_memory() -> tuple[int | None, int | None]:
    """Return Linux cgroup limit and usage when the runtime exposes them."""
    for limit_path, used_path in ((Path('/sys/fs/cgroup/memory.max'),
                                   Path('/sys/fs/cgroup/memory.current')),
                                  (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),
                                   Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))):
        try:
            limit = limit_path.read_text(encoding='ascii').strip()
            if limit != 'max':
                return int(limit), int(used_path.read_text(encoding='ascii').strip())
        except (OSError, ValueError):
            pass
    return None, None


def available_ram() -> int:
    limit, used = cgroup_memory()
    host = psutil.virtual_memory().available
    return min(host, max(0, limit-used)) if limit is not None and used is not None else host


def locate_inputs() -> tuple[Path, Path]:
    canonical = list(INPUT_ROOT.rglob('canonical_32000204.parquet'))
    cohorts = [p.parent for p in INPUT_ROOT.rglob('edges.parquet')
               if p.parent.name == 'edges_5000000' and (p.parent/'nodes.parquet').is_file()]
    if len(canonical) != 1 or len(cohorts) != 1:
        mounted = [str(p) for p in INPUT_ROOT.iterdir()] if INPUT_ROOT.is_dir() else []
        raise FileNotFoundError(f'Expected one canonical and one 5M cohort; '
                                f'found {len(canonical)} and {len(cohorts)}; mounts={mounted}')
    return canonical[0], cohorts[0]


def verify_inputs() -> dict:
    canonical, cohort_dir = locate_inputs()
    canonical_manifest = canonical.with_name('manifest.json')
    for path, expected in ((canonical, CANONICAL_SHA), (canonical_manifest, MANIFEST_SHA)):
        if file_sha256(path) != expected:
            raise ValueError(f'Input SHA-256 mismatch: {path.name}')
    source = json.loads(canonical_manifest.read_text(encoding='utf-8'))
    if (source['source_sha256'] != SOURCE_SHA or source['rows'] != 32_000_204 or
        source['prefixes']['32000204']['sha256'] != CANONICAL_SHA or
        source['prefixes']['32000204']['bytes'] != canonical.stat().st_size or
        source['transform']['order_by'] != ['event_ts', 'user_raw_id', 'movie_raw_id', 'source_row_id']):
        raise ValueError('Canonical provenance mismatch')
    cohort = json.loads((cohort_dir/'manifest.json').read_text(encoding='utf-8'))
    if (cohort['source_sha256'] != SOURCE_SHA or cohort['cutoff_ts'] != BASE_BOUNDARY or
        (cohort['positive_edges'], cohort['nodes'], cohort['users'], cohort['movies']) !=
        (4_999_981, 90_253, 49_587, 40_666)):
        raise ValueError('Approved cohort mismatch')
    for name in ('nodes', 'edges'):
        if file_sha256(cohort_dir/(name+'.parquet')) != cohort[name+'_sha256']:
            raise ValueError(f'Cohort {name} SHA-256 mismatch')
    with duckdb.connect(config={'memory_limit': '1GB', 'threads': '1'}) as con:
        count = con.execute('SELECT count(*) FROM read_parquet(?)', [str(canonical)]).fetchone()[0]
        if count != 32_000_204:
            raise ValueError('Canonical row count mismatch')
        for name, expected in (('nodes', 90_253), ('edges', 4_999_981)):
            observed = con.execute('SELECT count(*) FROM read_parquet(?)',
                                   [str(cohort_dir/(name+'.parquet'))]).fetchone()[0]
            if observed != expected:
                raise ValueError(f'Cohort {name} row count mismatch')
        schema = con.execute('DESCRIBE SELECT * FROM read_parquet(?)', [str(canonical)]).fetchall()
        if [(row[0], row[1]) for row in schema] != [
            ('event_id','BIGINT'), ('event_ts','BIGINT'), ('user_raw_id','BIGINT'),
            ('movie_raw_id','BIGINT'), ('rating','FLOAT'), ('source_row_id','BIGINT')]:
            raise ValueError('Canonical schema mismatch')
    return {'canonical_path':str(canonical), 'cohort_path':str(cohort_dir),
            'canonical_sha256': CANONICAL_SHA, 'canonical_manifest_sha256': MANIFEST_SHA,
            'raw_source_sha256': SOURCE_SHA, 'cohort_manifest_sha256': file_sha256(cohort_dir/'manifest.json'),
            'cohort_nodes_sha256': cohort['nodes_sha256'], 'cohort_edges_sha256': cohort['edges_sha256'],
            'canonical_rows': count, 'canonical_schema': [(row[0],row[1]) for row in schema]}


def worker(stage: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if stage == 'inputs':
        result = verify_inputs()
    else:
        # Every worker verifies the source again; no integrity check is skipped for speed.
        verified = verify_inputs()
        database = ROOT/'projection.duckdb'
        canonical = Path(verified['canonical_path'])
        cohort_dir = Path(verified['cohort_path'])
        graph = (ModelGraphReplay.create(database, cohort_dir, canonical) if stage == 'base'
                 else ModelGraphReplay(database))
        with graph:
            if stage == 'monthly':
                advance = graph.advance(MONTH_BOUNDARY)
                if (advance['counts']['rating_events'], advance['counts']['inserted']) != (125_929, 14_882):
                    raise ValueError('Monthly replay counts mismatch')
            record = graph.current()
            expected_version = 0 if stage == 'base' else 1
            expected_boundary = BASE_BOUNDARY if stage == 'base' else MONTH_BOUNDARY
            expected_edges = 4_999_981 if stage == 'base' else 5_014_863
            if (record['projection_version'] != expected_version or
                record['observed_before_ts'] != expected_boundary or
                record['edge_count'] != expected_edges or
                record['mapping_sha256'] != MAPPING_SHA or
                graph.con.execute('SELECT count(*) FROM nodes').fetchone()[0] != 90_253):
                raise ValueError('Projection state mismatch')
            if stage == 'base' and (record['edges_sha256'] != BASE_EDGES_SHA or
                                    record['degrees_sha256'] != BASE_DEGREES_SHA):
                raise ValueError('G0 edges/degrees mismatch')
            if record['last_canonical_order_key'][0] >= expected_boundary:
                raise ValueError('Future event in projection')
            snapshot = ROOT/(stage+'_snapshot')
            exported = graph.export_snapshot(snapshot, expected_version=expected_version)
            loaded, _, _ = load_projection_snapshot(snapshot, expected_mapping=MAPPING_SHA,
                expected_version=expected_version, expected_boundary=expected_boundary,
                expected_source=CANONICAL_SHA, materialize=False)
            if loaded != exported or exported['nnz'] != 2*expected_edges:
                raise ValueError('Snapshot reload mismatch')
            if exported['normalization_max_abs_diff'] > exported['normalization_atol']:
                raise ValueError('SQL/Python COO numerical mismatch')
            memory = graph.con.execute('SELECT * FROM duckdb_memory()').fetchall()
            result = {'inputs': verified, 'record': record, 'snapshot': exported,
                      'reload_verified': True, 'duckdb_memory': memory,
                      'future_edge_in_g0': False if stage == 'base' else None}
    (ROOT/(stage+'_result.json')).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cgroup_limit, _ = cgroup_memory()
    report = {'status': 'INCOMPLETE', 'git_head': os.environ.get('GOAL04_C3_GIT_HEAD'),
              'source_bundle_sha256': os.environ.get('GOAL04_C3_SOURCE_SHA256'),
              'environment': {'python': platform.python_version(),
              'duckdb': duckdb.__version__, 'cpu_count': os.cpu_count(),
              'physical_ram_bytes': psutil.virtual_memory().total,
              'cgroup_ram_limit_bytes': cgroup_limit,
              'visible_gpus': 0, 'gpu_names': []}, 'stages': [],
              'gpu_optimizer_steps': 0, 'quality_evaluations': 0}
    try:
        import torch
        report['environment']['torch'] = torch.__version__
        report['environment']['visible_gpus'] = torch.cuda.device_count()
        report['environment']['gpu_names'] = [torch.cuda.get_device_name(i)
                                               for i in range(torch.cuda.device_count())]
        report['environment']['gpu_memory_bytes'] = [torch.cuda.get_device_properties(i).total_memory
                                                     for i in range(torch.cuda.device_count())]
    except ImportError:
        report['environment']['torch'] = None
    target = ROOT/'report.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for stage in ('inputs','base','monthly'):
        began = time.monotonic()
        log = ROOT/(stage+'.log')
        with log.open('w', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable,'-m','scripts.kaggle_c3_runner','--worker',stage],
                                     cwd=Path(__file__).resolve().parents[1],
                                     stdout=stream,stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            peak_rss = peak_disk = peak_spill = 0
            min_available = available_ram()
            min_free_disk = shutil.disk_usage(ROOT).free
            reason = None
            while child.poll() is None:
                family = [process] + process.children(recursive=True)
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in family if p.is_running()))
                min_available = min(min_available, available_ram())
                min_free_disk = min(min_free_disk, shutil.disk_usage(ROOT).free)
                peak_disk = max(peak_disk, disk_bytes(ROOT))
                peak_spill = max(peak_spill, sum(disk_bytes(p) for p in ROOT.rglob('*')
                    if p.is_dir() and (p.name == 'tmp' or p.name.endswith('.tmp'))))
                if time.monotonic()-started > MAX_ELAPSED:
                    reason = 'PHASE_C_ELAPSED_BUDGET'
                elif min_available < 2*1024**3:
                    reason = 'CPU_AVAILABLE_BELOW_2_GIB'
                elif min_free_disk < 2*1024**3:
                    reason = 'DISK_FREE_BELOW_2_GIB'
                if reason:
                    for member in reversed(family):
                        try: member.terminate()
                        except psutil.NoSuchProcess: pass
                    break
                time.sleep(.2)
            code = child.wait()
        measured = {'stage':stage,'status':'PASS' if code == 0 and not reason else 'FAIL',
                    'exit_code':code,'stop_reason':reason,'wall_s':time.monotonic()-began,
                    'peak_rss_bytes':peak_rss,'min_available_ram_bytes':min_available,
                    'peak_disk_bytes':peak_disk,'min_free_disk_bytes':min_free_disk,
                    'sampled_peak_spill_bytes':peak_spill}
        report['stages'].append(measured)
        target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(measured),flush=True)
        if measured['status'] != 'PASS':
            report['status'] = 'FAIL'
            target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            raise SystemExit(1)
    report['status'] = 'PASS'
    report['elapsed_s'] = time.monotonic()-started
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','elapsed_s':report['elapsed_s']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('inputs','base','monthly'))
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
