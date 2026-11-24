"""One Kaggle cuda:0 LightGCN C0 trajectory; checkpoint candidates only."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import psutil
import torch
import duckdb

from scripts.kaggle_c3_runner import available_ram, cgroup_memory, disk_bytes, verify_inputs
from src.artifacts import file_sha256
from src.graph.equivalence import checksum
from src.models.c0 import C0Config, STEPS, load_candidate, load_g0, save_candidate
from src.models.lightgcn import LightGCN
from src.models.training import bpr_step, sample_batches


ROOT = Path('/kaggle/working/goal04_c4')
INPUT_ROOT = Path('/kaggle/input')
C3_SNAPSHOT_MANIFEST_SHA = '0feaa2c521b0adb364fa78468a86f8a39fac979d630053fe0cb8aff39910ebdb'
C3_REPORT_SHA = '29cc7adeb929b3b53a8ffdbb151c73acf28ba84430ff5d4026dca18ab224f26b'
SOURCE_SHA = '494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e'
MAPPING_SHA = '996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9'
BASE_EDGES_SHA = 'fb2c7fe7d64e018bebcd7101df41249dbfa57e1db5369fae01d023e9c2e5392a'
BASE_DEGREES_SHA = 'ee423151ba7070bc616b8a43475322410b759297c086b9331ba4f3318f49e2b4'
PRIOR_KAGGLE_ELAPSED_S = 404.0  # C3 failed mount run + successful preflight, including notebook overhead.
MAX_PHASE_C_S = 4 * 3600
MAX_OPTIMIZER_S = 3600


def locate_c3_snapshot() -> tuple[Path, dict]:
    candidates = [p for p in INPUT_ROOT.rglob('manifest.json') if p.parent.name == 'base_snapshot']
    matches = [p for p in candidates if file_sha256(p) == C3_SNAPSHOT_MANIFEST_SHA]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one approved C3 G0 snapshot; found {len(matches)}')
    snapshot = matches[0].parent
    report_path = snapshot.parent/'report.json'
    if file_sha256(report_path) != C3_REPORT_SHA:
        raise ValueError('C3 report file hash mismatch')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report['status'] != 'PASS' or report['gpu_optimizer_steps'] != 0 or
        report['quality_evaluations'] != 0):
        raise ValueError('C3 report is not a passing preflight')
    return snapshot, report


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def train_trajectory(state: dict, sparse: torch.Tensor, output: Path, *,
                     config: C0Config, provenance: dict, device: str,
                     steps: tuple[int, ...] = STEPS, toy_cpu: bool = False,
                     deadline: float | None = None) -> dict:
    """Save candidates from one model and one Adam optimizer; never evaluate quality."""
    if config != C0Config() or steps[0] != 0 or any(a >= b for a,b in zip(steps,steps[1:])):
        raise ValueError('C0 config or checkpoint schedule mismatch')
    if toy_cpu:
        if device != 'cpu' or steps[-1] > 5000:
            raise ValueError('Toy trajectory must stay on CPU')
    elif (device != 'cuda:0' or steps != STEPS or
          len(state['registry']) != 90_253 or len(state['edges']) != 4_999_981 or
          sum(row[1] == 'movie' for row in state['registry']) != 40_666):
        raise ValueError('Production C4 requires approved 5M graph and cuda:0')
    if sparse.layout != torch.sparse_coo or sparse.shape != (len(state['registry']),)*2:
        raise ValueError('Expected validated sparse G0 adjacency')
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'report.json'
    result = {'status':'INCOMPLETE','config':asdict(config),'provenance':provenance,
              'device':device,'checkpoints':[],'completed_step':0,
              'optimizer_elapsed_s':0.0,'selected_step':None,
              'validation_quality_evaluations':0}
    _write(report_path,result)
    torch.manual_seed(config.seed)
    if device == 'cuda:0':
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    model = LightGCN(len(state['registry']),config.dim,config.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
    sampler = sample_batches(state,config.batch_size,steps[-1],config.seed)
    optimizer_elapsed = 0.0
    latencies = []
    gpu_min_free_fraction = 1.0
    started = time.monotonic()

    def save(step: int, loss: float | None) -> None:
        path = output/f'candidate_{step}.pt'
        saved = save_candidate(path,model,step=step,config=config,
                               provenance=provenance,toy_cpu=toy_cpu)
        loaded = load_candidate(path,expected_sha256=saved['sha256'],
                                config=config,provenance=provenance,toy_cpu=toy_cpu)
        if not torch.equal(loaded.embedding.weight,model.embedding.weight.detach().cpu()):
            raise ValueError('Checkpoint reload tensor differs from live model')
        del loaded
        result['checkpoints'].append({'step':step,'file':path.name,
                                      'sha256':saved['sha256'],'bytes':path.stat().st_size,
                                      'training_loss':loss})
        _write(report_path,result)

    try:
        save(0,None)
        with (output/'steps.jsonl').open('w',encoding='utf-8') as log:
            for step in range(1,steps[-1]+1):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Phase C end-to-end elapsed cap')
                batch = next(sampler)
                if device == 'cuda:0': torch.cuda.synchronize(0)
                before = time.perf_counter()
                loss = bpr_step(model,sparse,optimizer,batch,device)
                if device == 'cuda:0': torch.cuda.synchronize(0)
                latency = time.perf_counter()-before
                optimizer_elapsed += latency
                latencies.append(latency)
                value = float(loss.item())
                log.write(json.dumps({'step':step,'loss':value,'optimizer_step_s':latency,
                                      'optimizer_elapsed_s':optimizer_elapsed})+'\n')
                result['completed_step'] = step
                result['optimizer_elapsed_s'] = optimizer_elapsed
                if optimizer_elapsed > MAX_OPTIMIZER_S:
                    raise TimeoutError('Optimizer elapsed cap exceeded')
                if step % 100 == 0 or step in steps:
                    log.flush()
                    if device == 'cuda:0':
                        free,total = torch.cuda.mem_get_info(0)
                        gpu_min_free_fraction = min(gpu_min_free_fraction,free/total)
                        if gpu_min_free_fraction < .30:
                            raise MemoryError('GPU effective headroom below 30%')
                    _write(report_path,result)
                if step in steps:
                    save(step,value)
        if [x['step'] for x in result['checkpoints']] != list(steps):
            raise ValueError('Missing candidate checkpoint')
        result.update(status='COMPLETE_CANDIDATES',trajectory_wall_s=time.monotonic()-started,
                      step_latency_s={'first':latencies[0],
                                      'median':statistics.median(latencies),
                                      'p95':sorted(latencies)[int(.95*(len(latencies)-1))],
                                      'max':max(latencies)},
                      gpu_min_free_fraction=gpu_min_free_fraction,
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if device=='cuda:0' else None,
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(0) if device=='cuda:0' else None,
                      gpu_free_end_bytes=torch.cuda.mem_get_info(0)[0] if device=='cuda:0' else None)
        _write(report_path,result)
        return result
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        _write(report_path,result)
        raise


def worker() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla T4':
        raise RuntimeError('Approved cuda:0 Tesla T4 unavailable')
    torch.set_num_threads(1)
    snapshot,c3_report = locate_c3_snapshot()
    inputs = verify_inputs()
    manifest = json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    if (manifest['source_sha256'] != inputs['canonical_sha256'] or
        manifest['cohort_manifest_sha256'] != inputs['cohort_manifest_sha256'] or
        manifest['mapping_sha256'] != MAPPING_SHA or
        manifest['edges_sha256'] != BASE_EDGES_SHA or
        manifest['degrees_sha256'] != BASE_DEGREES_SHA or
        manifest['edge_count'] != 4_999_981 or manifest['observed_before_ts'] != 1577836800):
        raise ValueError('G0 snapshot/source/cohort mismatch')
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(0)
    graph_manifest,state,sparse = load_g0(snapshot,source_sha256=SOURCE_SHA,
                                           mapping_sha256=MAPPING_SHA,device='cuda:0')
    if (len(state['registry']) != 90_253 or len(state['edges']) != 4_999_981 or
        checksum([row[:3] for row in state['registry']]) != MAPPING_SHA or
        checksum(state['edges']) != BASE_EDGES_SHA or
        graph_manifest['coo_sha256'] != manifest['coo_sha256']):
        raise ValueError('Loaded G0 state mismatch')
    load_elapsed = time.monotonic()-started
    gpu_free_after_load,gpu_total = torch.cuda.mem_get_info(0)
    if gpu_free_after_load/gpu_total < .30:
        raise MemoryError('GPU effective headroom below 30% after G0 load')
    provenance = {'mapping_sha256':MAPPING_SHA,
                  'cohort_manifest_sha256':inputs['cohort_manifest_sha256'],
                  'snapshot_manifest_sha256':C3_SNAPSHOT_MANIFEST_SHA,
                  'snapshot_coo_sha256':manifest['coo_sha256'],
                  'source_sha256':SOURCE_SHA,
                  'c3_report_sha256':C3_REPORT_SHA,
                  'git_head':os.environ['GOAL04_C4_GIT_HEAD'],
                  'source_bundle_sha256':os.environ['GOAL04_C4_SOURCE_SHA256'],
                  'base_cutoff_ts':1577836800,
                  'node_count':90_253,'base_edge_count':4_999_981,
                  'base_edges_sha256':BASE_EDGES_SHA,
                  'base_degrees_sha256':BASE_DEGREES_SHA}
    _write(ROOT/'inputs.json',{'verified_inputs':inputs,'c3_report_status':c3_report['status'],
                               'snapshot_manifest_sha256':C3_SNAPSHOT_MANIFEST_SHA,
                               'load_g0_elapsed_s':load_elapsed,'snapshot':manifest,
                               'gpu_peak_g0_load_allocated_bytes':torch.cuda.max_memory_allocated(0),
                               'gpu_peak_g0_load_reserved_bytes':torch.cuda.max_memory_reserved(0),
                               'gpu_free_after_g0_load_bytes':gpu_free_after_load,
                               'config':asdict(C0Config())})
    train_trajectory(state,sparse,ROOT/'candidates',config=C0Config(),
                     provenance=provenance,device='cuda:0',
                     deadline=time.monotonic()+MAX_PHASE_C_S-PRIOR_KAGGLE_ELAPSED_S-load_elapsed)


def run() -> None:
    ROOT.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    cgroup_limit,_ = cgroup_memory()
    report = {'status':'INCOMPLETE','git_head':os.environ['GOAL04_C4_GIT_HEAD'],
              'source_bundle_sha256':os.environ['GOAL04_C4_SOURCE_SHA256'],
              'prior_phase_c_kaggle_elapsed_s':PRIOR_KAGGLE_ELAPSED_S,
              'config':asdict(C0Config()),'device':'cuda:0',
              'environment':{'python':platform.python_version(),
                             'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
                             'duckdb':duckdb.__version__,
                             'visible_gpus':torch.cuda.device_count(),
                             'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             'cpu_count':os.cpu_count(),
                             'physical_ram_bytes':psutil.virtual_memory().total,
                             'cgroup_ram_limit_bytes':cgroup_limit},
              'selected_step':None,'validation_quality_evaluations':0}
    _write(ROOT/'report.json',report)
    with (ROOT/'worker.log').open('w',encoding='utf-8') as stream:
        child = subprocess.Popen([sys.executable,'-m','scripts.kaggle_c4_runner','--worker'],
                                 cwd=Path(__file__).resolve().parents[1],
                                 stdout=stream,stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        peak_rss = peak_disk = 0
        min_available = available_ram()
        min_free_disk = psutil.disk_usage(ROOT).free
        reason = None
        while child.poll() is None:
            try:
                family = [process] + process.children(recursive=True)
                peak_rss = max(peak_rss,sum(p.memory_info().rss for p in family if p.is_running()))
            except psutil.NoSuchProcess:
                family = [process]
            min_available = min(min_available,available_ram())
            min_free_disk = min(min_free_disk,psutil.disk_usage(ROOT).free)
            peak_disk = max(peak_disk,disk_bytes(ROOT))
            if time.monotonic()-started+PRIOR_KAGGLE_ELAPSED_S > MAX_PHASE_C_S:
                reason = 'PHASE_C_ELAPSED_CAP'
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
    report.update(elapsed_s=time.monotonic()-started,
                  cumulative_phase_c_kaggle_elapsed_s=time.monotonic()-started+PRIOR_KAGGLE_ELAPSED_S,
                  peak_process_tree_rss_bytes=peak_rss,
                  min_effective_available_ram_bytes=min_available,
                  peak_output_disk_bytes=peak_disk,min_free_disk_bytes=min_free_disk,
                  worker_exit_code=code,stop_reason=reason)
    worker_report = ROOT/'candidates/report.json'
    if worker_report.exists():
        report['trajectory'] = json.loads(worker_report.read_text(encoding='utf-8'))
    if code == 0 and not reason and report.get('trajectory',{}).get('status') == 'COMPLETE_CANDIDATES':
        report['status']='COMPLETE_CANDIDATES'
    _write(ROOT/'report.json',report)
    print(json.dumps({'status':report['status'],'elapsed_s':report['elapsed_s'],
                      'completed_step':report.get('trajectory',{}).get('completed_step',0)}),flush=True)
    if report['status'] != 'COMPLETE_CANDIDATES':
        raise SystemExit(1)


if __name__ == '__main__':
    if '--worker' in sys.argv: worker()
    else: run()
