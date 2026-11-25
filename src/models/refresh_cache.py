"""Content-addressed propagated embeddings; no checkpoint or optimizer writes."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from src.artifacts import file_sha256


SCHEMA = 'goal04_propagated_v1'
IDENTITY_FIELDS = frozenset({
    'strategy', 'checkpoint_sha256', 'snapshot_sha256', 'mapping_sha256',
    'source_sha256', 'cohort_sha256', 'projection_version', 'boundary',
    'edges_sha256', 'degrees_sha256', 'coo_sha256', 'node_count', 'dim',
    'layers', 'dtype', 'device', 'environment', 'repetition',
})


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False) + '\n').encode('utf-8')


def cache_key(identity: dict) -> str:
    if set(identity) != IDENTITY_FIELDS or identity['strategy'] not in ('S0', 'S1'):
        raise ValueError('Incomplete propagation identity')
    if ((identity['strategy'] == 'S0' and identity['repetition'] is not None) or
            (identity['strategy'] == 'S1' and identity['repetition'] not in (1, 2, 3))):
        raise ValueError('Invalid cache repetition scope')
    if any(type(identity[k]) is not int or identity[k] < 0 for k in
           ('projection_version', 'boundary', 'node_count', 'dim', 'layers')):
        raise ValueError('Invalid propagation shape/version')
    return hashlib.sha256(_canonical(identity)).hexdigest()


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous()
        digest.update(_canonical({'name': name, 'shape': list(array.shape),
                                  'dtype': str(array.dtype)}))
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def _check_tensor(tensor: torch.Tensor, identity: dict) -> None:
    if (tuple(tensor.shape) != (identity['node_count'], identity['dim']) or
            str(tensor.dtype) != identity['dtype'] or not torch.isfinite(tensor).all()):
        raise ValueError('Invalid propagated embeddings')


def publish_atomic(path: Path, embeddings: torch.Tensor, identity: dict,
                   parameter_before: str, parameter_after: str) -> dict:
    """Publish a fresh directory with manifest last; never replace an old cache."""
    path = Path(path)
    key = cache_key(identity)
    _check_tensor(embeddings, identity)
    if parameter_before != parameter_after:
        raise ValueError('C0 parameters changed')
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=path.name + '.partial-', dir=path.parent))
    try:
        payload = temp / 'embeddings.pt'
        torch.save(embeddings.detach().cpu().clone(), payload)
        manifest = dict(schema_version=SCHEMA, cache_key=key, identity=identity,
                        parameter_before=parameter_before, parameter_after=parameter_after,
                        embeddings_sha256=file_sha256(payload),
                        shape=list(embeddings.shape), dtype=str(embeddings.dtype),
                        inference_mode=True, forward_attempts=1, optimizer_updates=0)
        with payload.open('r+b') as handle:
            handle.flush()
            os.fsync(handle.fileno())
        with (temp / 'manifest.json').open('wb') as handle:
            handle.write(_canonical(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        temp.rename(path)
        return manifest
    except Exception:
        shutil.rmtree(temp)
        raise


def load_verified(path: Path, identity: dict, *, expected_manifest_sha256: str | None = None
                  ) -> tuple[torch.Tensor, dict]:
    if not expected_manifest_sha256 or len(expected_manifest_sha256) != 64:
        raise ValueError('external manifest SHA-256 required')
    path = Path(path)
    key = cache_key(identity)
    raw = (path / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError('Cache manifest hash mismatch')
    manifest = json.loads(raw)
    if (raw != _canonical(manifest) or set(manifest) != {
            'schema_version', 'cache_key', 'identity', 'parameter_before',
            'parameter_after', 'embeddings_sha256', 'shape', 'dtype',
            'inference_mode', 'forward_attempts', 'optimizer_updates'} or
            manifest['schema_version'] != SCHEMA or manifest['cache_key'] != key or
            manifest['identity'] != identity or
            manifest['parameter_before'] != manifest['parameter_after'] or
            manifest['inference_mode'] is not True or
            manifest['forward_attempts'] != 1 or manifest['optimizer_updates'] != 0):
        raise ValueError('Incompatible cache manifest')
    payload = path / 'embeddings.pt'
    if file_sha256(payload) != manifest['embeddings_sha256']:
        raise ValueError('Cache tensor hash mismatch')
    tensor = torch.load(payload, map_location='cpu', weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError('Cache payload is not a tensor')
    _check_tensor(tensor, identity)
    if manifest['shape'] != list(tensor.shape) or manifest['dtype'] != str(tensor.dtype):
        raise ValueError('Cache tensor metadata mismatch')
    return tensor, manifest
