"""Build a private D4 preflight package that cannot run without Owner metadata."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

from scripts.prepare_kaggle_c3 import SOURCES as C3_SOURCES
from src.artifacts import file_sha256


ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = 'khoakp/moviegraph-v2-d4-resource-preflight'
SOURCES = tuple(dict.fromkeys((*C3_SOURCES,
    'scripts/kaggle_d4_runner.py', 'src/benchmarking/common.py',
    'src/evaluation/baselines.py', 'src/evaluation/c0_validation.py',
    'src/evaluation/checkpoint_selection.py', 'src/evaluation/refresh_boundary.py',
    'src/evaluation/refresh_strategies.py', 'src/evaluation/temporal.py',
    'src/models/c0.py', 'src/models/lightgcn.py', 'src/models/refresh_cache.py',
    'src/models/training.py', 'results/goal04_training/selection.json')))


def prepare(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    content = io.BytesIO()
    hashes = {}
    with zipfile.ZipFile(content, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            raw = (ROOT / name).read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    bundle = content.getvalue()
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    script = (
        '"""Private D4 preflight; approval gate runs before GPU or data work."""\n'
        'import base64, hashlib, os, runpy, sys, zipfile\n'
        'from pathlib import Path\n'
        f'payload = base64.b64decode({base64.b64encode(bundle).decode()!r})\n'
        f'assert hashlib.sha256(payload).hexdigest() == {bundle_sha!r}\n'
        "root = Path('/kaggle/working/goal04_d4_source')\n"
        "root.mkdir(parents=True, exist_ok=False)\n"
        "archive = root.with_suffix('.zip')\n"
        "archive.write_bytes(payload)\n"
        "with zipfile.ZipFile(archive) as z: z.extractall(root)\n"
        f"os.environ['GOAL04_D4_BUNDLE_SHA256'] = {bundle_sha!r}\n"
        f"os.environ['GOAL04_D4_GIT_HEAD'] = {git_head!r}\n"
        "sys.path.insert(0, str(root))\n"
        "runpy.run_module('scripts.kaggle_d4_runner', run_name='__main__')\n"
    )
    (output / 'runner.py').write_text(script, encoding='utf-8')
    metadata = {'id': KERNEL_ID, 'title': 'MovieGraph v2 D4 Resource Preflight',
                'code_file': 'runner.py', 'language': 'python', 'kernel_type': 'script',
                'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': ['khoakp/moviegraph-v2-feasibility-input',
                                    'khoakp/moviegraph-v2-c3-canonical'],
                'kernel_sources': ['khoakp/moviegraph-v2-c3-real-5m-preflight',
                                   'khoakp/moviegraph-v2-c4-single-t4-training'],
                'competition_sources': [], 'model_sources': []}
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n',
                                                 encoding='utf-8')
    manifest = {'schema_version': 'goal04_d4_package_v1', 'status': 'BLOCKED_OWNER_APPROVAL',
                'kernel_id': KERNEL_ID, 'git_head': git_head,
                'source_bundle_sha256': bundle_sha, 'source_sha256': hashes,
                'runner_sha256': file_sha256(output / 'runner.py'),
                'selection_sha256': hashes['results/goal04_training/selection.json'],
                'boundary': 1609459200, 'device': 'cuda:0',
                'proposed_forward_attempts': 2, 'optimizer_updates': 0,
                'owner_decision_reference': None,
                'private_input_sources': metadata['dataset_sources'] + metadata['kernel_sources']}
    (output / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                                  encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2))
