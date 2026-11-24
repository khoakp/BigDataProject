"""Independently verify the official C5 output without selecting C0."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from src.artifacts import file_sha256
from src.evaluation.c0_validation import boundaries
from src.models.c0 import STEPS


def verify(root: Path, package: Path, c4_root: Path) -> dict:
    root, package, c4_root = Path(root), Path(package), Path(c4_root)
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    manifest = json.loads((package/'package-manifest.json').read_text(encoding='utf-8'))
    c4 = json.loads((c4_root/'candidates/report.json').read_text(encoding='utf-8'))
    rows = json.loads((root/'validation-matrix.json').read_text(encoding='utf-8'))
    if (report['status'] != 'COMPLETE_VALIDATION' or
        'selected_step' in report or
        report['validation_matrix'] != rows or len(rows) != 60 or
        report['propagation_forwards'] != 5 or report['optimizer_updates'] != 0 or
        report['source_bundle_sha256'] != manifest['source_bundle_sha256'] or
        report['git_head'] != manifest['git_head'] or
        report['budget_audit_sha256'] != manifest['budget_audit_sha256'] or
        report['prior_verified_kaggle_elapsed_s'] != manifest['verified_prior_phase_c_kaggle_elapsed_s'] or
        report['worker_exit_code'] != 0 or report['stop_reason'] is not None or
        report['cumulative_phase_c_kaggle_elapsed_s'] >= 14400-120 or
        not math.isclose(report['cumulative_phase_c_kaggle_elapsed_s'],
                         report['prior_verified_kaggle_elapsed_s']+report['job_elapsed_s'],
                         abs_tol=1e-5)):
        raise ValueError('C5 status, provenance, budget or forward count mismatch')
    if (report['inputs']['c4_verification_status'] != 'PASS' or
        report['inputs']['c4_report_sha256'] != file_sha256(c4_root/'report.json') or
        report['inputs']['catalog_movies'] != 40_666 or
        report['inputs']['windows'] != 12 or
        report['inputs']['mapping_sha256'] != c4['provenance']['mapping_sha256'] or
        report['inputs']['canonical_sha256'] != c4['provenance']['source_sha256'] or
        report['inputs']['base_edges_sha256'] != c4['provenance']['base_edges_sha256']):
        raise ValueError('C5 input integrity mismatch')
    if (file_sha256(root/'validation-matrix.json') !=
        report['artifact_sha256']['validation-matrix.json'] or
        file_sha256(root/'candidate-comparison.csv') !=
        report['artifact_sha256']['candidate-comparison.csv']):
        raise ValueError('C5 output file hash mismatch')
    candidates = report['candidate_scores']
    if ([row['step'] for row in candidates] != list(STEPS) or
        [row['checkpoint_sha256'] for row in candidates] !=
        [row['sha256'] for row in c4['checkpoints']]):
        raise ValueError('C5 candidate universe mismatch')
    cuts = boundaries()
    seen = set()
    for row in rows:
        key = (row['step'],row['window_index'])
        if key in seen or row['step'] not in STEPS or row['window_index'] not in range(12):
            raise ValueError('Duplicate or unexpected candidate/window')
        seen.add(key)
        index = row['window_index']
        if (row['start_ts'] != cuts[index] or row['end_exclusive_ts'] != cuts[index+1] or
            row['mask_boundary_ts'] != cuts[index] or
            row['mask_policy'] != 'rolling_observed_positive_before_boundary' or
            row['mask_base_edges_sha256'] != c4['provenance']['base_edges_sha256'] or
            row['mask_canonical_source_sha256'] != c4['provenance']['source_sha256'] or
            row['catalog_mapping_sha256'] != c4['provenance']['mapping_sha256'] or
            row['catalog_movies'] != 40_666 or
            row['eligible_users'] <= 0 or row['eligible_labels'] <= 0 or
            row['gpu_mode'] not in ('dual_t4','single_t4_fallback') or
            row['numerical_equivalence'] not in ('PASS','FAIL_FALLBACK','ONE_GPU')):
            raise ValueError('C5 window boundary, mask or catalog mismatch')
        for metric in ('recall_20','ndcg_20'):
            if not isinstance(row[metric],(float,int)) or not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError('C5 quality metric invalid')
    if seen != {(step,index) for step in STEPS for index in range(12)}:
        raise ValueError('C5 matrix is incomplete')
    for candidate in candidates:
        subset = [row for row in rows if row['step'] == candidate['step']]
        if (candidate['window_count'] != 12 or
            candidate['gpu_mode'] not in ('dual_t4','single_t4_fallback') or
            any(row['gpu_mode'] != candidate['gpu_mode'] or
                row['numerical_equivalence'] != candidate['numerical_equivalence']['status']
                for row in subset)):
            raise ValueError('C5 candidate mode or window count mismatch')
        for metric,field in (('recall_20','validation_recall_20'),
                             ('ndcg_20','validation_ndcg_20')):
            if not math.isclose(candidate[field],sum(row[metric] for row in subset)/12,
                                abs_tol=1e-12):
                raise ValueError('C5 macro-window aggregation mismatch')
    with (root/'candidate-comparison.csv').open(newline='',encoding='utf-8') as stream:
        comparison = list(csv.DictReader(stream))
    if ([int(row['step']) for row in comparison] != list(STEPS) or
        len(comparison) != 5 or any(
            not math.isclose(float(row['validation_ndcg_20']),candidate['validation_ndcg_20'],abs_tol=1e-12)
            for row,candidate in zip(comparison,candidates))):
        raise ValueError('C5 comparison table mismatch')
    return {'status':'PASS','matrix_rows':60,'candidate_steps':list(STEPS),
            'propagation_forwards':5,'optimizer_updates':0,
            'selected_step':None,'job_elapsed_s':report['job_elapsed_s'],
            'cumulative_phase_c_kaggle_elapsed_s':report['cumulative_phase_c_kaggle_elapsed_s'],
            'artifact_sha256':report['artifact_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--c4-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = verify(args.root,args.package,args.c4_root)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'])
