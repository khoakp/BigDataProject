"""Prepare a private C5 validation package; draft cannot run without budget audit."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile

from scripts.prepare_kaggle_c4 import ROOT, SOURCES as C4_SOURCES
from src.artifacts import file_sha256


SOURCES = tuple(dict.fromkeys((*C4_SOURCES,
    'scripts/verify_kaggle_c4.py',
    'scripts/kaggle_c5_runner.py',
    'results/goal04_c4/run_04/package-manifest.json',
    'results/goal04_protocol/monthly.csv')))
KERNEL_ID = 'khoakp/moviegraph-v2-c5-2020-validation'


def prepare(output: Path, *, budget_audit: Path | None = None) -> dict:
    verified_prior_seconds = None
    audit_sha256 = None
    if budget_audit is not None:
        audit = json.loads(budget_audit.read_text(encoding='utf-8'))
        if audit.get('status') not in ('VERIFIED','OWNER_APPROVED_ACCOUNTING'):
            raise ValueError('Phase C Kaggle budget audit is incomplete')
        if (audit['status'] == 'OWNER_APPROVED_ACCOUNTING' and
            not audit.get('owner_decision_reference')):
            raise ValueError('Owner budget accounting decision is missing')
        verified_prior_seconds = audit.get('cumulative_phase_c_kaggle_elapsed_s')
        if (isinstance(verified_prior_seconds,bool) or
            not isinstance(verified_prior_seconds,(int,float)) or
            not math.isfinite(verified_prior_seconds) or
            not 0 <= verified_prior_seconds < 4*3600 or
            not math.isclose(audit.get('remaining_budget_s',-1),
                             4*3600-verified_prior_seconds,abs_tol=1e-6)):
            raise ValueError('Invalid verified Phase C Kaggle elapsed')
        audit_sha256 = file_sha256(budget_audit)
    output.mkdir(parents=True, exist_ok=False)
    payload = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in SOURCES:
            content = (ROOT/name).read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,content,compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    bundle = payload.getvalue()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    git_head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    budget_line = (f"os.environ['GOAL04_VERIFIED_PRIOR_PHASE_C_S'] = {str(verified_prior_seconds)!r}\n"
                   if verified_prior_seconds is not None else
                   "raise RuntimeError('C5 submission blocked: Phase C Kaggle elapsed not audited')\n")
    if audit_sha256 is not None:
        budget_line += f"os.environ['GOAL04_C5_BUDGET_AUDIT_SHA256'] = {audit_sha256!r}\n"
    script = (
        '"""Private C5 validation; no optimizer, no holdout."""\n'
        'import base64, hashlib, os, runpy, sys, time, zipfile\n'
        'from pathlib import Path\n'
        "os.environ['GOAL04_C5_JOB_STARTED_MONOTONIC'] = str(time.monotonic())\n"
        + budget_line +
        f"payload = base64.b64decode({base64.b64encode(bundle).decode('ascii')!r})\n"
        f"assert hashlib.sha256(payload).hexdigest() == {bundle_hash!r}\n"
        "root = Path('/kaggle/working/goal04_c5_source')\n"
        "root.mkdir(parents=True, exist_ok=False)\n"
        "archive_path = root.with_suffix('.zip')\n"
        "archive_path.write_bytes(payload)\n"
        "with zipfile.ZipFile(archive_path) as archive: archive.extractall(root)\n"
        f"os.environ['GOAL04_C5_SOURCE_SHA256'] = {bundle_hash!r}\n"
        f"os.environ['GOAL04_C5_GIT_HEAD'] = {git_head!r}\n"
        "sys.path.insert(0, str(root))\n"
        "runpy.run_module('scripts.kaggle_c5_runner', run_name='__main__')\n"
    )
    (output/'runner.py').write_text(script,encoding='utf-8')
    metadata = {'id':KERNEL_ID,'title':'MovieGraph v2 C5 2020 Validation',
                'code_file':'runner.py','language':'python','kernel_type':'script',
                'is_private':True,'enable_gpu':True,'enable_internet':False,
                'machine_shape':'NvidiaTeslaT4',
                'dataset_sources':['khoakp/moviegraph-v2-feasibility-input',
                                   'khoakp/moviegraph-v2-c3-canonical'],
                'kernel_sources':['khoakp/moviegraph-v2-c3-real-5m-preflight',
                                  'khoakp/moviegraph-v2-c4-single-t4-training'],
                'competition_sources':[],'model_sources':[]}
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    manifest = {'kernel_id':KERNEL_ID,'git_head':git_head,
                'source_sha256':hashes,'source_bundle_sha256':bundle_hash,
                'runner_sha256':file_sha256(output/'runner.py'),
                'status':'BLOCKED_BUDGET_AUDIT' if verified_prior_seconds is None else 'READY',
                'verified_prior_phase_c_kaggle_elapsed_s':verified_prior_seconds,
                'budget_audit_sha256':audit_sha256,
                'dataset_sources':metadata['dataset_sources'],
                'kernel_sources':metadata['kernel_sources'],
                'candidate_steps':[0,500,1500,3000,5000],
                'validation_windows':12,'catalog_movies':40_666,
                'propagation_forward_limit':5,'optimizer_update_limit':0}
    (output/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-audit',type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output,budget_audit=args.budget_audit),indent=2))
