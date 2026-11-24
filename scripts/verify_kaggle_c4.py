"""Verify the one C4 trajectory's downloaded checkpoints and resource report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.artifacts import file_sha256
from src.models.c0 import C0Config, STEPS, load_candidate


def verify(root: Path, package: Path) -> dict:
    root = Path(root)
    package_manifest = json.loads((Path(package)/'package-manifest.json').read_text(encoding='utf-8'))
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    trajectory = json.loads((root/'candidates'/'report.json').read_text(encoding='utf-8'))
    inputs = json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    config = C0Config()
    expected_config = package_manifest['config']
    if (report['status'] != 'COMPLETE_CANDIDATES' or
        trajectory['status'] != 'COMPLETE_CANDIDATES' or
        report['trajectory'] != trajectory or
        report['source_bundle_sha256'] != package_manifest['source_bundle_sha256'] or
        report['git_head'] != package_manifest['git_head'] or
        report['device'] != 'cuda:0' or trajectory['device'] != 'cuda:0' or
        report['environment']['gpu_names'][0] != 'Tesla T4' or
        report['cumulative_phase_c_kaggle_elapsed_s'] > 4*3600 or
        trajectory['optimizer_elapsed_s'] > 3600 or
        report['selected_step'] is not None or trajectory['selected_step'] is not None or
        report['validation_quality_evaluations'] != 0 or
        trajectory['validation_quality_evaluations'] != 0 or
        trajectory['completed_step'] != 5000 or
        expected_config != {'dim':config.dim,'layers':config.layers,
                            'batch_size':config.batch_size,'lr':config.lr,
                            'seed':config.seed,'steps':list(config.steps),
                            'device':'cuda:0'}):
        raise ValueError('C4 status, config, device or budget mismatch')
    provenance = trajectory['provenance']
    if (provenance['node_count'] != 90_253 or
        provenance['base_edge_count'] != 4_999_981 or
        provenance['mapping_sha256'] != '996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9' or
        provenance['base_edges_sha256'] != 'fb2c7fe7d64e018bebcd7101df41249dbfa57e1db5369fae01d023e9c2e5392a' or
        provenance['snapshot_manifest_sha256'] != package_manifest['c3_snapshot_manifest_sha256'] or
        provenance['c3_report_sha256'] != package_manifest['c3_report_sha256'] or
        provenance['source_sha256'] != inputs['verified_inputs']['canonical_sha256'] or
        inputs['snapshot']['coo_sha256'] != provenance['snapshot_coo_sha256'] or
        inputs['snapshot']['observed_before_ts'] != 1577836800):
        raise ValueError('C4 G0 provenance mismatch')
    checkpoint_rows = trajectory['checkpoints']
    if [row['step'] for row in checkpoint_rows] != list(STEPS):
        raise ValueError('Missing, extra or unordered checkpoints')
    log_rows = [json.loads(line) for line in (root/'candidates'/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(log_rows) != 5000 or [row['step'] for row in log_rows] != list(range(1,5001)):
        raise ValueError('Training step log is incomplete')
    elapsed = 0.0
    for row in log_rows:
        if not math.isfinite(row['loss']) or row['optimizer_step_s'] <= 0:
            raise ValueError('Invalid step loss or latency')
        elapsed += row['optimizer_step_s']
        if not math.isclose(elapsed,row['optimizer_elapsed_s'],abs_tol=1e-5):
            raise ValueError('Optimizer elapsed trajectory mismatch')
    if not math.isclose(elapsed,trajectory['optimizer_elapsed_s'],abs_tol=1e-5):
        raise ValueError('Final optimizer elapsed mismatch')
    verified = []
    for row in checkpoint_rows:
        path = root/'candidates'/row['file']
        if (path.name != f"candidate_{row['step']}.pt" or
            path.stat().st_size != row['bytes'] or
            file_sha256(path) != row['sha256']):
            raise ValueError(f"Checkpoint file mismatch at step {row['step']}")
        model = load_candidate(path,expected_sha256=row['sha256'],config=config,
                               provenance=provenance)
        if tuple(model.embedding.weight.shape) != (90_253,64) or model.layers != 3:
            raise ValueError('Reloaded checkpoint model shape mismatch')
        if row['step'] != 0 and not math.isclose(row['training_loss'],
                            log_rows[row['step']-1]['loss'],abs_tol=1e-12):
            raise ValueError('Checkpoint loss does not match step log')
        verified.append({'step':row['step'],'sha256':row['sha256'],'bytes':row['bytes']})
    return {'status':'PASS','git_head':report['git_head'],
            'source_bundle_sha256':report['source_bundle_sha256'],
            'device':'cuda:0','optimizer_steps':5000,
            'optimizer_elapsed_s':elapsed,
            'cumulative_phase_c_kaggle_elapsed_s':report['cumulative_phase_c_kaggle_elapsed_s'],
            'selected_step':None,'validation_quality_evaluations':0,
            'checkpoints':verified}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = verify(args.root,args.package)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'])
