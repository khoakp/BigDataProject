"""Build a private Kaggle C3 kernel with current source embedded in its code file."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from src.artifacts import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'scripts/kaggle_c3_runner.py',
    'src/artifacts.py',
    'src/graph/equivalence.py',
    'src/graph/model_replay.py',
    'src/graph/normalize.py',
    'src/ingest/canonical.py',
    'src/ingest/replay.py',
)
KERNEL_ID = 'khoakp/moviegraph-v2-c3-real-5m-preflight'


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    payload = io.BytesIO()
    source_hashes = {}
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in SOURCES:
            path = ROOT/name
            content = path.read_bytes()
            source_hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2020,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
    bundle = payload.getvalue()
    bundle_hash = hashlib.sha256(bundle).hexdigest()
    git_head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    script = (
        '"""Private Kaggle C3 preflight: bundled project code, no model training."""\n'
        'import base64, hashlib, os, runpy, sys, zipfile\n'
        'from pathlib import Path\n'
        f"payload = base64.b64decode({base64.b64encode(bundle).decode('ascii')!r})\n"
        f"assert hashlib.sha256(payload).hexdigest() == {bundle_hash!r}\n"
        "root = Path('/kaggle/working/goal04_c3_source')\n"
        "root.mkdir(parents=True, exist_ok=False)\n"
        "archive_path = root.with_suffix('.zip')\n"
        "archive_path.write_bytes(payload)\n"
        "with zipfile.ZipFile(archive_path) as archive: archive.extractall(root)\n"
        f"os.environ['GOAL04_C3_SOURCE_SHA256'] = {bundle_hash!r}\n"
        f"os.environ['GOAL04_C3_GIT_HEAD'] = {git_head!r}\n"
        "sys.path.insert(0, str(root))\n"
        "runpy.run_module('scripts.kaggle_c3_runner', run_name='__main__')\n"
    )
    (output/'runner.py').write_text(script,encoding='utf-8')
    metadata = {'id':KERNEL_ID, 'title':'MovieGraph v2 C3 Real 5M Preflight',
                'code_file':'runner.py','language':'python','kernel_type':'script',
                'is_private':True,'enable_gpu':True,'enable_internet':False,
                'machine_shape':'NvidiaTeslaT4',
                'dataset_sources':['khoakp/moviegraph-v2-feasibility-input',
                                   'khoakp/moviegraph-v2-c3-canonical'],
                'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    manifest = {'kernel_id':KERNEL_ID,'git_head':git_head,'source_sha256':source_hashes,
                'source_bundle_sha256':bundle_hash,'runner_sha256':file_sha256(output/'runner.py'),
                'dataset_sources':metadata['dataset_sources'],
                'optimizer_steps':0,'quality_evaluations':0}
    (output/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
