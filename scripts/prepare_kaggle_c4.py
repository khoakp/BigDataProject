"""Package one private Kaggle C4 training kernel from the current v2 source."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from scripts.prepare_kaggle_c3 import ROOT, SOURCES as C3_SOURCES
from src.artifacts import file_sha256


SOURCES = tuple(dict.fromkeys((*C3_SOURCES,
    'scripts/kaggle_c4_runner.py',
    'src/evaluation/c0_validation.py',
    'src/evaluation/checkpoint_selection.py',
    'src/evaluation/temporal.py',
    'src/models/c0.py',
    'src/models/lightgcn.py',
    'src/models/training.py')))
KERNEL_ID = 'khoakp/moviegraph-v2-c4-single-t4-training'


def prepare(output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    payload = io.BytesIO()
    source_hashes = {}
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in SOURCES:
            content = (ROOT/name).read_bytes()
            source_hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info,content,compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    bundle = payload.getvalue()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    git_head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    script = (
        '"""Private C4 single-T4 LightGCN training; no validation quality."""\n'
        'import base64, hashlib, os, runpy, sys, zipfile\n'
        'from pathlib import Path\n'
        f"payload = base64.b64decode({base64.b64encode(bundle).decode('ascii')!r})\n"
        f"assert hashlib.sha256(payload).hexdigest() == {bundle_hash!r}\n"
        "root = Path('/kaggle/working/goal04_c4_source')\n"
        "root.mkdir(parents=True, exist_ok=False)\n"
        "archive_path = root.with_suffix('.zip')\n"
        "archive_path.write_bytes(payload)\n"
        "with zipfile.ZipFile(archive_path) as archive: archive.extractall(root)\n"
        f"os.environ['GOAL04_C4_SOURCE_SHA256'] = {bundle_hash!r}\n"
        f"os.environ['GOAL04_C4_GIT_HEAD'] = {git_head!r}\n"
        "sys.path.insert(0, str(root))\n"
        "runpy.run_module('scripts.kaggle_c4_runner', run_name='__main__')\n"
    )
    (output/'runner.py').write_text(script,encoding='utf-8')
    metadata = {'id':KERNEL_ID,'title':'MovieGraph v2 C4 Single T4 Training',
                'code_file':'runner.py','language':'python','kernel_type':'script',
                'is_private':True,'enable_gpu':True,'enable_internet':False,
                'machine_shape':'NvidiaTeslaT4',
                'dataset_sources':['khoakp/moviegraph-v2-feasibility-input',
                                   'khoakp/moviegraph-v2-c3-canonical'],
                'kernel_sources':['khoakp/moviegraph-v2-c3-real-5m-preflight'],
                'competition_sources':[],'model_sources':[]}
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    manifest = {'kernel_id':KERNEL_ID,'git_head':git_head,
                'source_sha256':source_hashes,'source_bundle_sha256':bundle_hash,
                'runner_sha256':file_sha256(output/'runner.py'),
                'c3_snapshot_manifest_sha256':'0feaa2c521b0adb364fa78468a86f8a39fac979d630053fe0cb8aff39910ebdb',
                'c3_report_sha256':'29cc7adeb929b3b53a8ffdbb151c73acf28ba84430ff5d4026dca18ab224f26b',
                'dataset_sources':metadata['dataset_sources'],
                'kernel_sources':metadata['kernel_sources'],
                'config':{'dim':64,'layers':3,'batch_size':1024,'lr':.001,
                          'seed':7,'steps':[0,500,1500,3000,5000],
                          'device':'cuda:0'},
                'validation_quality_evaluations':0}
    (output/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
