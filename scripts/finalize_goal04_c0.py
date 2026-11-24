"""Audit C5 validation and lock the selected C4 checkpoint by reference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from scripts.verify_kaggle_c3 import verify as verify_c3
from scripts.verify_kaggle_c4 import verify as verify_c4
from scripts.verify_kaggle_c5 import verify as verify_c5
from src.artifacts import file_sha256
from src.evaluation.c0_validation import boundaries
from src.evaluation.checkpoint_selection import select_checkpoint
from src.models.c0 import C0Config, SCHEMA, STEPS, load_candidate


ROOT = Path(__file__).resolve().parents[1]
C3 = ROOT/'results/goal04_c3/run_03'
C4 = ROOT/'results/goal04_c4/run_04'
C5 = ROOT/'results/goal04_c5/run_02'
C5_PACKAGE = ROOT/'data/processed/goal04_c5_retry_official_v4'
COMMON_FIELDS = ('start_utc','end_utc','start_ts','end_exclusive_ts',
                 'eligible_users','eligible_labels','catalog_movies',
                 'catalog_mapping_sha256','mask_policy','mask_boundary_ts',
                 'mask_base_edges_sha256','mask_canonical_source_sha256')


def recompute_matrix(rows: list[dict], comparison: list[dict], provenance: dict) -> list[dict]:
    """Check the full candidate/window grid before using any quality metric."""
    cuts = boundaries()
    if len(rows) != len(STEPS)*12 or len(cuts) != 13:
        raise ValueError('Missing validation row')
    by_key = {}
    for row in rows:
        step, index = row['step'], row['window_index']
        if step not in STEPS or type(index) is not int or index not in range(12) or (step,index) in by_key:
            raise ValueError('Unexpected or duplicate step/window')
        if (row['start_ts'] != cuts[index] or row['end_exclusive_ts'] != cuts[index+1] or
            row['start_utc'] != datetime.fromtimestamp(cuts[index],timezone.utc).isoformat() or
            row['end_utc'] != datetime.fromtimestamp(cuts[index+1],timezone.utc).isoformat() or
            row['mask_boundary_ts'] != cuts[index] or
            row['catalog_movies'] != 40_666 or
            row['catalog_mapping_sha256'] != provenance['mapping_sha256'] or
            row['mask_policy'] != 'rolling_observed_positive_before_boundary' or
            row['mask_base_edges_sha256'] != provenance['base_edges_sha256'] or
            row['mask_canonical_source_sha256'] != provenance['source_sha256'] or
            row['eligible_users'] <= 0 or row['eligible_labels'] <= 0 or
            row['numerical_equivalence'] not in ('PASS','FAIL_FALLBACK','ONE_GPU')):
            raise ValueError('Boundary, mask, catalog or source mismatch')
        for name in ('recall_20','ndcg_20'):
            value = row[name]
            if type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid validation metric')
        by_key[step,index] = row
    if set(by_key) != {(step,index) for step in STEPS for index in range(12)}:
        raise ValueError('Missing validation row')
    for index in range(12):
        reference = tuple(by_key[STEPS[0],index][field] for field in COMMON_FIELDS)
        if any(tuple(by_key[step,index][field] for field in COMMON_FIELDS) != reference
               for step in STEPS[1:]):
            raise ValueError('Candidate user/label counts or mask/catalog provenance differs')
    if [int(row['step']) for row in comparison] != list(STEPS):
        raise ValueError('Candidate comparison step mismatch')
    records = []
    for step, official in zip(STEPS,comparison):
        group = [by_key[step,index] for index in range(12)]
        recall = sum(row['recall_20'] for row in group)/12
        ndcg = sum(row['ndcg_20'] for row in group)/12
        if (int(official['window_count']) != 12 or
            not math.isclose(recall,float(official['validation_recall_20']),abs_tol=1e-12) or
            not math.isclose(ndcg,float(official['validation_ndcg_20']),abs_tol=1e-12)):
            raise ValueError('Official comparison differs from monthly macro metrics')
        records.append({'step':step,'validation_recall_20':recall,
                        'validation_ndcg_20':ndcg})
    return records


def verify_selected_checkpoint(path: Path, *, expected_sha256: str, expected_step: int,
                               config: C0Config, provenance: dict) -> dict:
    """Strictly reload original C4 bytes and require the selected payload step."""
    if path.name != f'candidate_{expected_step}.pt':
        raise ValueError('Selected checkpoint filename/step mismatch')
    model = load_candidate(path,expected_sha256=expected_sha256,
                           config=config,provenance=provenance)
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if (payload['step'] != expected_step or
        tuple(model.embedding.weight.shape) != (provenance['node_count'],config.dim) or
        model.layers != config.layers):
        raise ValueError('Selected checkpoint step or tensor shape mismatch')
    return {'sha256':file_sha256(path),'bytes':path.stat().st_size,
            'payload_step':payload['step'],'schema_version':payload['schema_version'],
            'embedding_shape':list(model.embedding.weight.shape),'layers':model.layers}


def build_selection_manifest() -> dict:
    """Recompute the selection from immutable C3/C4/C5 evidence."""
    c3 = verify_c3(C3/'goal04_c3',C3)
    c4 = verify_c4(C4/'goal04_c4',C4)
    c5 = verify_c5(C5/'download/goal04_c5',C5_PACKAGE,C4/'goal04_c4')
    if any(item['status'] != 'PASS' for item in (c3,c4,c5)):
        raise ValueError('C3/C4/C5 evidence verification failed')
    matrix_path = C5/'download/goal04_c5/validation-matrix.json'
    comparison_path = C5/'download/goal04_c5/candidate-comparison.csv'
    c5_report_path = C5/'download/goal04_c5/report.json'
    c5_assessment = json.loads((C5/'assessment.json').read_text(encoding='utf-8'))
    c5_report = json.loads(c5_report_path.read_text(encoding='utf-8'))
    c5_recorded_verification = json.loads((C5/'verification.json').read_text(encoding='utf-8'))
    c4_recorded_verification = json.loads((C4/'verification.json').read_text(encoding='utf-8'))
    c4_report_path = C4/'goal04_c4/candidates/report.json'
    c4_report = json.loads(c4_report_path.read_text(encoding='utf-8'))
    provenance = c4_report['provenance']
    if (file_sha256(matrix_path) != c5_assessment['validation_matrix_sha256'] or
        file_sha256(comparison_path) != c5_assessment['comparison_sha256'] or
        file_sha256(c5_report_path) != c5_assessment['report_sha256'] or
        c5_assessment['status'] != 'COMPLETE_VALIDATION' or
        c5_assessment['source_bundle_sha256'] != c5_report['source_bundle_sha256'] or
        c5_recorded_verification['status'] != 'PASS' or
        c5_recorded_verification['matrix_rows'] != 60 or
        c5_recorded_verification['selected_step'] is not None or
        c5_recorded_verification['artifact_sha256'] != c5['artifact_sha256'] or
        c4_recorded_verification['status'] != 'PASS' or
        c4_recorded_verification['checkpoints'] != c4['checkpoints'] or
        'KernelWorkerStatus.COMPLETE' not in (C5/'status.txt').read_text(encoding='utf-8') or
        c5['selected_step'] is not None or
        provenance['mapping_sha256'] != c3['snapshots']['base']['mapping_sha256'] or
        provenance['source_sha256'] != c3['inputs']['canonical_sha256'] or
        provenance['cohort_manifest_sha256'] != c3['inputs']['cohort_manifest_sha256'] or
        provenance['base_edges_sha256'] != c3['snapshots']['base']['edges_sha256'] or
        provenance['snapshot_coo_sha256'] != c3['snapshots']['base']['coo_sha256'] or
        provenance['snapshot_manifest_sha256'] !=
        file_sha256(C3/'goal04_c3/base_snapshot/manifest.json') or
        provenance['node_count'] != 90_253 or
        provenance['base_edge_count'] != 4_999_981 or
        [row['step'] for row in c5_report['candidate_scores']] != list(STEPS) or
        any(row['gpu_mode'] != 'dual_t4' or
            row['numerical_equivalence']['status'] != 'PASS' or
            row['numerical_equivalence']['sampled_users'] != 16 or
            row['numerical_equivalence']['max_abs_top20_score_diff'] != 0
            for row in c5_report['candidate_scores'])):
        raise ValueError('Final evidence SHA-256, status, mapping or source mismatch')
    rows = json.loads(matrix_path.read_text(encoding='utf-8'))
    with comparison_path.open(newline='',encoding='utf-8') as stream:
        comparison = list(csv.DictReader(stream))
    records = recompute_matrix(rows,comparison,provenance)
    selection = select_checkpoint(records)
    if selection['step'] != 1500:
        raise ValueError('Independent selection differs from expected step 1500; stop for Owner review')
    checkpoint_row = next(row for row in c4_report['checkpoints'] if row['step']==selection['step'])
    checkpoint_path = C4/'goal04_c4/candidates'/checkpoint_row['file']
    config = C0Config()
    if c4_report['config'] != {
        'dim':config.dim,'layers':config.layers,'batch_size':config.batch_size,
        'lr':config.lr,'seed':config.seed,'steps':list(config.steps)}:
        raise ValueError('C4 config differs from approved C0 workload')
    loaded = verify_selected_checkpoint(checkpoint_path,
        expected_sha256=checkpoint_row['sha256'],expected_step=selection['step'],
        config=config,provenance=provenance)
    if loaded['schema_version'] != SCHEMA or loaded['bytes'] != checkpoint_row['bytes']:
        raise ValueError('Selected checkpoint format or file size mismatch')
    manifest = {
        'schema_version':'goal04_c0_selection_v1','status':'PHASE_C_COMPLETE',
        'selected_step':selection['step'],
        'selection_rule':['max_macro_window_ndcg_20','max_macro_window_recall_20',
                          'min_optimizer_step'],
        'validation_year':2020,'validation_windows':12,'catalog_movies':40_666,
        'candidate_scores':records,
        'checkpoint_path':checkpoint_path.relative_to(ROOT).as_posix(),
        'checkpoint':loaded,'config':c4_report['config'],
        'g0_snapshot_path':(C3/'goal04_c3/base_snapshot').relative_to(ROOT).as_posix(),
        'g0_snapshot_manifest_sha256':provenance['snapshot_manifest_sha256'],
        'mapping_sha256':provenance['mapping_sha256'],
        'base_edges_sha256':provenance['base_edges_sha256'],
        'snapshot_coo_sha256':provenance['snapshot_coo_sha256'],
        'canonical_source_sha256':provenance['source_sha256'],
        'cohort_manifest_sha256':provenance['cohort_manifest_sha256'],
        'training_git_head':provenance['git_head'],
        'training_source_bundle_sha256':provenance['source_bundle_sha256'],
        'validation_source_bundle_sha256':c5_assessment['source_bundle_sha256'],
        'evidence_sha256':{
            'c3_verification':file_sha256(C3/'verification.json'),
            'c4_verification':file_sha256(C4/'verification.json'),
            'c5_verification':file_sha256(C5/'verification.json'),
            'c5_matrix':file_sha256(matrix_path),
            'c5_comparison':file_sha256(comparison_path),
            'c5_report':file_sha256(c5_report_path)},
        'budget_accounting':{
            'hard_phase_c_s':14_400,
            'charged_through_c5_last_log_s':c5_assessment['owner_scope_charged_through_last_log_s'],
            'remaining_on_last_log_basis_s':c5_assessment['remaining_budget_on_last_log_basis_s'],
            'terminal_elapsed_exact':False},
        'limitations':['single_seed_7','fixed_warm_cohort',
                       'stale_g0_checkpoint_selection_bias',
                       'no_independent_holdout_quality','no_per_user_gate_proof']}
    return manifest


def finalize(output: Path) -> dict:
    """Write one selection manifest; never copy or rewrite checkpoint bytes."""
    if output.exists():
        raise FileExistsError(f'Selection manifest already exists: {output}')
    manifest = build_selection_manifest()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def verify_selection(path: Path) -> dict:
    """Re-audit the source evidence and compare the locked selection record."""
    expected = build_selection_manifest()
    actual = json.loads(path.read_text(encoding='utf-8'))
    if actual != expected:
        raise ValueError('C0 selection manifest differs from verified C3/C4/C5 evidence')
    return {'status':'PASS','selected_step':expected['selected_step'],
            'selection_manifest_sha256':file_sha256(path),
            'checkpoint_sha256':expected['checkpoint']['sha256'],
            'evidence_sha256':expected['evidence_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/goal04_training/selection.json')
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_selection(args.output) if args.verify else finalize(args.output),
                     indent=2,ensure_ascii=False))
