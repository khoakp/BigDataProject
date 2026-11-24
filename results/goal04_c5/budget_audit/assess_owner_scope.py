"""Account C3 v3 and C4 v4 under the Owner's 24 Nov scope decision."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
LIMIT = 4 * 3600


def main():
    evidence = json.loads((ROOT/'api-evidence.json').read_text(encoding='utf-8'))
    listing = (ROOT/'latest-kernels-list.csv').read_text(encoding='utf-8-sig').splitlines()
    last_run_time = {parts[0]: parts[3] for line in listing
                     if (parts := line.split(',')) and len(parts) >= 4}
    selected = [('c3',3,'moviegraph-v2-c3-real-5m-preflight'),
                ('c4',4,'moviegraph-v2-c4-single-t4-training')]
    rows = []
    for phase, version, slug in selected:
        item = next(r for r in evidence if (r['phase'],r['version']) == (phase,version))
        if item['status'] != 'KernelWorkerStatus.COMPLETE':
            raise ValueError(f'{phase} v{version} is not complete')
        log = json.loads(item['log'])
        last_log = max(entry['time'] for entry in log)
        ref = f'khoakp/{slug}'
        start = datetime.fromisoformat(last_run_time[ref]).replace(tzinfo=timezone.utc)
        report_path = (REPO/'results/goal04_c3/run_03/goal04_c3/report.json'
                       if phase == 'c3' else
                       REPO/'results/goal04_c4/run_04/goal04_c4/report.json')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['elapsed_s'] <= 0 or last_log < report['elapsed_s']:
            raise ValueError('Kaggle log does not cover runner elapsed')
        rows.append({'phase':phase,'version':version,'status':item['status'],
                     'kaggle_list_last_run_time_utc':start.isoformat(),
                     'last_log_observed_utc_inferred':(start+timedelta(seconds=last_log)).isoformat(),
                     'runner_elapsed_s':report['elapsed_s'],
                     'last_log_relative_s':last_log,
                     'charged_elapsed_s':last_log,
                     'terminal_job_elapsed_s':None})
    charged = sum(row['charged_elapsed_s'] for row in rows)
    if not 0 < charged < LIMIT:
        raise ValueError('No Phase C budget remains under Owner scope')
    result = {'status':'OWNER_APPROVED_ACCOUNTING',
              'owner_decision_reference':'Owner chat 2026-11-24: count C3 v3 and C4 v4 only; earlier queued/canceled save versions excluded',
              'accounting_scope':'C3 v3 and C4 v4 only',
              'hard_budget_s':LIMIT,'included_versions':rows,
              'runner_subtotal_s':sum(row['runner_elapsed_s'] for row in rows),
              'cumulative_phase_c_kaggle_elapsed_s':charged,
              'remaining_budget_s':LIMIT-charged,
              'codex_preparation_elapsed_s':'excluded; local work, not a Kaggle job',
              'measurement_limit':'Charged elapsed reaches each version’s final Kaggle log event, including notebook rendering. Terminal teardown after the last log is not exposed by the API; this is an Owner-scope accounting figure, not an exact terminal-duration claim.',
              'excluded_version_note':'Raw seven-version evidence remains in api-evidence.json. C3 v2 has ERROR and a 26.90 s log, so it is excluded by the Owner scope decision, not reclassified as a zero-execution queued job.',
              'submission_gate':'Owner scope budget accounting complete; C5 official submit remains a separate stop gate after local preparation report.'}
    target = ROOT/'assessment-owner-scope.json'
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'],round(charged,3),round(result['remaining_budget_s'],3))


if __name__ == '__main__':
    main()
