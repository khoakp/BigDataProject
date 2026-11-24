"""Guarded local real-5M CPU preflight. Never upload data or run a model."""
import argparse
import gc
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
from src.ingest.d02_profile import run as canonicalize


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'results/model_cohort_v2_final/edges_5000000'
RAW = ROOT.parent / 'ml-32m/ratings.csv'


def disk_bytes(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(base)/name).stat().st_size
            except FileNotFoundError:
                pass  # Spill files can disappear while sampling.
    return total


def verify_inputs():
    approved = json.loads((ROOT/'results/model_cohort_v2_manifest.json').read_text())
    cohort = json.loads((COHORT/'manifest.json').read_text())
    expected = next(r for r in approved['cohorts'] if r['edge_budget']==5_000_000)
    assert (cohort['positive_edges'],cohort['nodes'],cohort['users'],cohort['movies']) == (4_999_981,90_253,49_587,40_666)
    assert cohort['cutoff_ts']==1577836800 and cohort['selection']=='user_recency'
    assert file_sha256(RAW)==cohort['source_sha256']==approved['source_sha256']
    for name in ('nodes','edges'):
        assert file_sha256(COHORT/(name+'.parquet'))==cohort[name+'_sha256']==expected[name+'_sha256']
    return cohort


def worker(stage, artifacts, report, canonical_input=None):
    report.mkdir(parents=True,exist_ok=True)
    cohort = verify_inputs()
    canonical_dir = artifacts/'canonical'
    source = canonical_dir/'canonical_32000204.parquet'
    if stage=='canonical':
        if canonical_input is None:
            manifest = canonicalize(RAW,canonical_dir,max_prefix=32_000_204,memory_limit='1GB')
        else:
            manifest=json.loads((canonical_input/'manifest.json').read_text())
            original=canonical_input/source.name
            assert file_sha256(original)==manifest['prefixes']['32000204']['sha256']
            canonical_dir.mkdir(parents=True,exist_ok=False)
            shutil.copy2(original,source)
            shutil.copy2(canonical_input/'manifest.json',canonical_dir/'manifest.json')
            assert file_sha256(source)==manifest['prefixes']['32000204']['sha256']
        assert manifest['rows']==32_000_204 and manifest['source_sha256']==cohort['source_sha256']
        result = dict(source_sha256=manifest['source_sha256'],canonical=manifest['prefixes']['32000204'],
                      operation='reuse_verified_canonical' if canonical_input else 'canonicalize_raw')
    else:
        canonical_manifest = json.loads((canonical_dir/'manifest.json').read_text())
        assert canonical_manifest['source_sha256']==cohort['source_sha256']
        assert file_sha256(source)==canonical_manifest['prefixes']['32000204']['sha256']
        database = artifacts/'projection.duckdb'
        graph = (ModelGraphReplay.create(database,COHORT,source) if stage=='base'
                 else ModelGraphReplay(database))
        with graph:
            graph.con.execute("PRAGMA enable_profiling='json'")
            graph.con.execute('SET profiling_output=?',[str(report/(stage+'_duckdb.json'))])
            if stage=='monthly':
                record=graph.advance(1580515200)  # All January events, no February.
                assert record['counts']['inserted']==14_882
                assert record['counts']['rating_events']==125_929
            record=graph.current()
            assert record['edge_count']==(4_999_981 if stage=='base' else 5_014_863)
            assert graph.con.execute('SELECT count(*) FROM nodes').fetchone()[0]==90_253
            snapshot=artifacts/(stage+'_snapshot')
            exported=graph.export_snapshot(snapshot,expected_version=record['projection_version'])
            gc.collect()
            loaded,state,adj=load_projection_snapshot(snapshot,
                expected_mapping=record['mapping_sha256'],expected_version=record['projection_version'],
                expected_boundary=record['observed_before_ts'],expected_source=file_sha256(source),materialize=False)
            assert loaded['edge_count']==record['edge_count'] and loaded['nnz']==2*record['edge_count']
            result=dict(record=record,snapshot=exported,reload_verified=True,
                        duckdb_memory=graph.con.execute('SELECT * FROM duckdb_memory()').fetchall())
            graph.con.execute('SELECT count(*) FROM edges').fetchall()
    (report/(stage+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')


def guard(stage, artifacts, report, *, wall_cap, canonical_input=None):
    """Only terminate our own child when a declared resource cap is crossed."""
    started=time.perf_counter()
    peak_rss=0
    minimum_available=psutil.virtual_memory().available
    peak_disk=0
    peak_spill=0
    reason=None
    with (report/(stage+'.log')).open('w',encoding='utf-8') as log:
        command=[sys.executable,'-m','scripts.goal04_preflight','worker',
                 '--stage',stage,'--artifacts',str(artifacts),'--report',str(report)]
        if canonical_input:
            command.extend(['--canonical-input',str(canonical_input)])
        child=subprocess.Popen(command,
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid)
        while child.poll() is None:
            available=psutil.virtual_memory().available
            minimum_available=min(minimum_available,available)
            try:
                # Windows venv python.exe is a launcher; include the actual worker.
                family=[process,*process.children(recursive=True)]
                peak_rss=max(peak_rss,sum(p.memory_info().rss for p in family if p.is_running()))
            except psutil.NoSuchProcess:
                break
            peak_disk=max(peak_disk,disk_bytes(artifacts))
            peak_spill=max(peak_spill,sum(disk_bytes(p) for p in artifacts.rglob('*')
                                          if p.is_dir() and (p.name=='tmp' or p.name.endswith('.tmp'))))
            if available < 2*1024**3:
                reason='CPU_AVAILABLE_BELOW_2_GIB'
            elif time.perf_counter()-started > wall_cap:
                reason='WALL_TIME_CAP'
            if reason:
                for descendant in reversed(family[1:]):
                    try:
                        descendant.terminate()
                    except psutil.NoSuchProcess:
                        pass
                child.terminate()
                break
            time.sleep(.1)
        code=child.wait()
    result=dict(stage=stage,status='PASS' if code==0 and reason is None else 'STOPPED',
                exit_code=code,stop_reason=reason,wall_s=time.perf_counter()-started,
                wall_cap_s=wall_cap,peak_rss_bytes=peak_rss,min_host_available_bytes=minimum_available,
                peak_disk_bytes=peak_disk,final_disk_bytes=disk_bytes(artifacts),
                sampled_peak_spill_bytes=peak_spill,sample_interval_s=.1,duckdb_limit='1GB')
    (report/(stage+'_resources.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def run(artifacts, report, canonical_input=None):
    artifacts.mkdir(parents=True,exist_ok=False)
    report.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                environment=dict(python=platform.python_version(),duckdb=duckdb.__version__,physical_ram=psutil.virtual_memory().total),
                stages=[],gpu_runs=0,quality_evaluations=0,
                source_sha256={p:file_sha256(ROOT/p) for p in
                    ('scripts/goal04_preflight.py','src/graph/model_replay.py','src/graph/normalize.py','src/ingest/d02_profile.py','src/ingest/canonical.py')})
    target=report/'report.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    for stage,cap in (('canonical',1800),('base',300),('monthly',300)):
        measured=guard(stage,artifacts,report,wall_cap=cap,canonical_input=canonical_input)
        result['stages'].append(measured)
        if measured['status']!='PASS':
            result['status']='PREFLIGHT_STOPPED_NO_GPU_AUTHORIZATION'
            target.write_text(json.dumps(result,indent=2)+'\n')
            return result
        target.write_text(json.dumps(result,indent=2)+'\n')
    result['status']='PASS'
    target.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','worker'])
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--stage',choices=['canonical','base','monthly'])
    parser.add_argument('--canonical-input',type=Path,help='Previously verified canonical directory, copied into fresh output')
    args=parser.parse_args()
    if args.mode=='worker':
        worker(args.stage,args.artifacts,args.report,args.canonical_input)
    else:
        result=run(args.artifacts,args.report,args.canonical_input)
        sys.exit(0 if result['status']=='PASS' else 1)
