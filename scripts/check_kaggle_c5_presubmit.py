"""Local, read-only gate for the one approved C5 Kaggle submission."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from src.artifacts import file_sha256
from src.evaluation.c0_validation import boundaries


def check(package: Path, audit: Path, evidence_root: Path) -> dict:
    package, audit, evidence_root = map(Path,(package,audit,evidence_root))
    manifest = json.loads((package/'package-manifest.json').read_text(encoding='utf-8'))
    accounting = json.loads(audit.read_text(encoding='utf-8'))
    c3 = json.loads((evidence_root/'presubmit-c3-verification.json').read_text())
    c4 = json.loads((evidence_root/'presubmit-c4-verification.json').read_text())
    inputs = json.loads((evidence_root/'input-integrity.json').read_text())
    if (manifest['status'] != 'READY' or
        manifest['budget_audit_sha256'] != file_sha256(audit) or
        manifest['verified_prior_phase_c_kaggle_elapsed_s'] !=
        accounting['cumulative_phase_c_kaggle_elapsed_s'] or
        accounting['remaining_budget_s'] < 120 or
        accounting['hard_budget_s'] != 14400):
        raise ValueError('Budget or input assessment mismatch')
    if any(value['status'] != 'PASS' for value in (c3,c4,inputs)):
        raise ValueError('C3, C4 or local input verification failed')
    if (c4['optimizer_steps'] != 5000 or
        [r['step'] for r in c4['checkpoints']] != [0,500,1500,3000,5000] or
        inputs['catalog_movies'] != 40_666 or inputs['nodes'] != 90_253 or
        inputs['snapshot_edge_count'] != 4_999_981 or
        inputs['mapping_sha256'] != c3['snapshots']['base']['mapping_sha256']):
        raise ValueError('Candidate or graph contract mismatch')
    metadata = json.loads((package/'kernel-metadata.json').read_text(encoding='utf-8'))
    if (metadata['id'] != manifest['kernel_id'] or
        metadata['id'] != 'khoakp/moviegraph-v2-c5-2020-validation' or
        not metadata['is_private'] or
        not metadata['enable_gpu'] or metadata['enable_internet'] or
        metadata['machine_shape'] != 'NvidiaTeslaT4' or
        metadata['kernel_sources'] != manifest['kernel_sources'] or
        metadata['dataset_sources'] != manifest['dataset_sources']):
        raise ValueError('Kaggle privacy, GPU or input metadata mismatch')
    script = (package/'runner.py').read_text(encoding='utf-8')
    if file_sha256(package/'runner.py') != manifest['runner_sha256']:
        raise ValueError('Kaggle runner file hash mismatch')
    parsed = ast.parse(script)
    budget_assignments = [node.value for node in parsed.body
                          if isinstance(node,ast.Assign) and
                          any(isinstance(target,ast.Subscript) and
                              isinstance(target.value,ast.Attribute) and
                              isinstance(target.value.value,ast.Name) and
                              target.value.value.id == 'os' and
                              target.value.attr == 'environ' and
                              isinstance(target.slice,ast.Constant) and
                              target.slice.value == 'GOAL04_VERIFIED_PRIOR_PHASE_C_S'
                              for target in node.targets)]
    if (len(budget_assignments) != 1 or
        not isinstance(budget_assignments[0],ast.Constant) or
        not isinstance(budget_assignments[0].value,str) or
        budget_assignments[0].value != str(accounting['cumulative_phase_c_kaggle_elapsed_s'])):
        raise ValueError('C5 wrapper budget environment value must be a string')
    payload = next(node.value for node in parsed.body if isinstance(node,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id == 'payload' for t in node.targets))
    bundle = base64.b64decode(ast.literal_eval(payload.args[0]))
    if hashlib.sha256(bundle).hexdigest() != manifest['source_bundle_sha256']:
        raise ValueError('Kaggle source bundle hash mismatch')
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(manifest['source_sha256']):
            raise ValueError('Kaggle source bundle is incomplete')
        for name, expected in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Embedded source hash mismatch: {name}')
        c5_source = archive.read('scripts/kaggle_c5_runner.py').decode('utf-8')
        with archive.open('results/goal04_protocol/monthly.csv') as source:
            monthly = list(csv.DictReader(io.TextIOWrapper(source,encoding='utf-8')))
    cuts = boundaries()
    rows = [row for row in monthly if row['window'].startswith('2020-')]
    if (len(cuts) != 13 or len(rows) != 12 or
        [(int(r['boundary_ts']),int(r['end_exclusive_ts'])) for r in rows] !=
        list(zip(cuts,cuts[1:]))):
        raise ValueError('Approved twelve-month 2020 validation calendar mismatch')
    if (any(token in c5_source for token in ('select_checkpoint','selected_step',
                                           'bpr_step','torch.optim')) or
        c5_source.count('model(sparse)') != 1 or
        'FINALIZATION_MARGIN_S = 120' not in c5_source or
        "'--worker' in sys.argv" not in c5_source or
        "'validation_matrix'" not in c5_source or
        'len(report[\'validation_matrix\']) != 60' not in c5_source):
        raise ValueError('C5 selection, optimizer, forward or runtime guard mismatch')
    if ("GOAL04_C5_JOB_STARTED_MONOTONIC" not in script or
        str(accounting['cumulative_phase_c_kaggle_elapsed_s']) not in script or
        manifest['budget_audit_sha256'] not in script):
        raise ValueError('C5 wrapper budget guard mismatch')
    return {'status':'PASS','package_manifest_sha256':file_sha256(package/'package-manifest.json'),
            'runner_sha256':manifest['runner_sha256'],
            'source_bundle_sha256':manifest['source_bundle_sha256'],
            'budget_audit_sha256':manifest['budget_audit_sha256'],
            'c3_verification':'PASS','c4_verification':'PASS','input_integrity':'PASS',
            'windows':12,'catalog_movies':40_666,'candidate_steps':[0,500,1500,3000,5000],
            'optimizer_updates':0,'maximum_propagation_forwards':5,
            'selected_c0':False,'finalization_margin_s':120}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = check(args.package,args.audit,args.evidence_root)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'])
