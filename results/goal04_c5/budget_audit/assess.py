"""Summarize saved Kaggle API evidence without inventing missing elapsed."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
LIMIT = 4 * 3600


def main():
    evidence = json.loads((ROOT/'api-evidence.json').read_text(encoding='utf-8'))
    rows = []
    lower_bound = 0.0
    for item in evidence:
        log = json.loads(item['log'] or '[]')
        last_log_s = max((entry.get('time', 0) for entry in log), default=None)
        if last_log_s is not None:
            lower_bound += last_log_s
        dates = sorted({f['creation_date'] for f in item['files'] if f['name'] == 'report.json'})
        rows.append({'phase':item['phase'],'version':item['version'],
                     'status':item['status'],
                     'start_utc':None,'end_utc':None,'elapsed_s':None,
                     'report_file_creation_utc_minute':dates[0] if dates else None,
                     'last_log_relative_s':last_log_s,
                     'metadata_last_run_time_untrusted':item['metadata_last_run_time'],
                     'reason_elapsed_unverified':'Kaggle API has no version start/end or duration; canceled jobs have no log'
                         if last_log_s is None else 'Log timestamp is a lower bound, not terminal job duration'})
    c3 = json.loads((REPO/'results/goal04_c3/run_03/goal04_c3/report.json').read_text())
    c4 = json.loads((REPO/'results/goal04_c4/run_04/goal04_c4/report.json').read_text())
    latest = (ROOT/'latest-kernels-list.csv').read_text(encoding='utf-8-sig').splitlines()
    repeated_c4_v4 = (ROOT/'metadata-repeat-c4-v4.txt').read_text(encoding='utf-8-sig').strip()
    output = {'status':'INCOMPLETE','hard_budget_s':LIMIT,
              'versions':rows,
              'latest_kernel_list_rows':latest,
              'repeated_c4_v4_metadata_last_run_time':repeated_c4_v4,
              'verified_runner_elapsed_s':{'c3_v3':c3['elapsed_s'],'c4_v4':c4['elapsed_s']},
              'verified_runner_subtotal_s':c3['elapsed_s']+c4['elapsed_s'],
              'observed_log_lower_bound_s':lower_bound,
              'remaining_budget_s':None,
              'maximum_possible_remaining_from_log_lower_bound_s':LIMIT-lower_bound,
              'codex_preparation_elapsed_s':'excluded from Kaggle job budget',
              'metadata_warning':'Version metadata last_run_time says 21 Nov while report files were created 23 Nov; C4 v4 changed from the first saved 04:29:50 to the repeated 04:40:43, so it cannot establish version start/end.',
              'submission_gate':'BLOCKED: Owner decision required because cumulative Kaggle elapsed is unverified'}
    (ROOT/'assessment.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(output['status'], 'lower bound', round(lower_bound,2), 'remaining unknown')


if __name__ == '__main__':
    main()
