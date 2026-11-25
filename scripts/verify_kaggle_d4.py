"""Static presubmit validation; never submits a kernel or launches CUDA."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from src.artifacts import file_sha256


def verify(package: Path) -> dict:
    package = Path(package)
    manifest = json.loads((package / 'package-manifest.json').read_text(encoding='utf-8'))
    metadata = json.loads((package / 'kernel-metadata.json').read_text(encoding='utf-8'))
    script = (package / 'runner.py').read_text(encoding='utf-8')
    if (manifest['status'] != 'BLOCKED_OWNER_APPROVAL' or
            manifest['owner_decision_reference'] is not None or
            manifest['runner_sha256'] != file_sha256(package / 'runner.py') or
            metadata['is_private'] is not True or metadata['enable_internet'] is not False or
            metadata['enable_gpu'] is not True or metadata['machine_shape'] != 'NvidiaTeslaT4' or
            metadata['code_file'] != 'runner.py' or
            manifest['boundary'] != 1609459200 or
            manifest['proposed_forward_attempts'] != 2):
        raise ValueError('D4 package metadata or approval gate mismatch')
    match = re.search(r"payload = base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", script)
    if not match:
        raise ValueError('Embedded source bundle missing')
    bundle = base64.b64decode(match.group(1), validate=True)
    if hashlib.sha256(bundle).hexdigest() != manifest['source_bundle_sha256']:
        raise ValueError('Embedded source bundle SHA mismatch')
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        if set(archive.namelist()) != set(manifest['source_sha256']):
            raise ValueError('Source inventory mismatch')
        for name, expected in manifest['source_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Source SHA mismatch: {name}')
        runner = archive.read('scripts/kaggle_d4_runner.py').decode('utf-8')
        if 'approval_gate(' not in runner or "if __name__ == '__main__':" not in runner:
            raise ValueError('Approval gate source missing')
    return {'status': 'PASS_BLOCKED_PENDING_OWNER',
            'source_bundle_sha256': manifest['source_bundle_sha256'],
            'source_files': len(manifest['source_sha256']),
            'runner_sha256': manifest['runner_sha256']}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().package), indent=2))
