"""One LightGCN C0 trajectory with strict, versioned candidate artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch

from src.artifacts import file_sha256
from src.graph.equivalence import checksum
from src.graph.model_replay import load_projection_snapshot
from src.evaluation.c0_validation import aggregate_validation, evaluate_embeddings
from src.evaluation.checkpoint_selection import select_checkpoint
from src.models.lightgcn import LightGCN
from src.models.lightgcn import to_sparse_tensor
from src.models.training import bpr_step, sample_batches


STEPS = (0, 500, 1500, 3000, 5000)
SCHEMA = 'lightgcn_v2_c0_candidate_v1'


@dataclass(frozen=True)
class C0Config:
    dim: int = 64
    layers: int = 3
    batch_size: int = 1024
    lr: float = .001
    seed: int = 7
    steps: tuple[int, ...] = STEPS


def _verify_config(config):
    if config != C0Config():
        raise ValueError('C0 config differs from approved v2 workload')


def load_g0(snapshot: Path, *, source_sha256: str, mapping_sha256: str,
            device='cpu'):
    """Require a verified base snapshot; never train from a replayed 2020 graph."""
    manifest,state,adjacency = load_projection_snapshot(
        snapshot,expected_mapping=mapping_sha256,expected_version=0,
        expected_boundary=1577836800,expected_source=source_sha256)
    if manifest['edge_count'] != len(state['edges']):
        raise ValueError('Base snapshot edge count mismatch')
    return manifest,state,to_sparse_tensor(adjacency,torch.device(device))


def save_candidate(path: Path, model: LightGCN, *, step: int,
                   config: C0Config, provenance: dict, toy_cpu: bool = False) -> dict:
    _verify_config(config)
    if (step not in config.steps and not (toy_cpu and 0 <= step <= config.steps[-1])) or path.exists():
        raise ValueError('Unexpected checkpoint step or existing path')
    required = ('mapping_sha256', 'cohort_manifest_sha256', 'snapshot_coo_sha256',
                'source_sha256', 'git_head', 'base_cutoff_ts', 'node_count',
                'base_edge_count', 'base_edges_sha256')
    if any(not provenance.get(key) for key in required) or provenance['base_cutoff_ts'] != 1577836800:
        raise ValueError('Incomplete or wrong G0 provenance')
    if model.layers != config.layers or tuple(model.embedding.weight.shape) != (provenance['node_count'],config.dim):
        raise ValueError('Model shape mismatch')
    payload = dict(schema_version=SCHEMA, step=step, config=asdict(config),
                   provenance=provenance,
                   state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    torch.save(payload,path)
    return dict(path=str(path),sha256=file_sha256(path),step=step)


def load_candidate(path: Path, *, expected_sha256: str, config: C0Config,
                   provenance: dict, device='cpu', toy_cpu: bool = False) -> LightGCN:
    _verify_config(config)
    if file_sha256(path) != expected_sha256:
        raise ValueError('Checkpoint file hash mismatch')
    data = torch.load(path,map_location='cpu',weights_only=True)
    if (set(data) != {'schema_version','step','config','provenance','state_dict'} or
        data['schema_version'] != SCHEMA or
        (data['step'] not in config.steps and not (toy_cpu and 0 <= data['step'] <= config.steps[-1])) or
        data['config'] != asdict(config) or data['provenance'] != provenance):
        raise ValueError('Checkpoint metadata mismatch')
    model = LightGCN(provenance['node_count'],config.dim,config.layers)
    if set(data['state_dict']) != set(model.state_dict()) or any(
        data['state_dict'][key].shape != value.shape or
        data['state_dict'][key].dtype != value.dtype
        for key,value in model.state_dict().items()):
        raise ValueError('Checkpoint tensor mismatch')
    model.load_state_dict(data['state_dict'],strict=True)
    return model.to(device).eval()


def train_candidates(state, adjacency, windows, output: Path, *,
                     config: C0Config, provenance: dict, device='cpu',
                     steps=None):
    """One optimizer trajectory; reduced steps allowed only for toy CPU tests."""
    _verify_config(config)
    if len(state['registry']) != provenance['node_count']:
        raise ValueError('Graph/mapping node count mismatch')
    if checksum([(node,kind,raw) for node,kind,raw,*_ in state['registry']]) != provenance['mapping_sha256']:
        raise ValueError('Graph/mapping hash mismatch')
    if adjacency.shape != (provenance['node_count'],)*2:
        raise ValueError('Snapshot shape mismatch')
    if len(state['edges']) != provenance.get('base_edge_count'):
        raise ValueError('G0 edge count mismatch')
    if checksum(state['edges']) != provenance.get('base_edges_sha256'):
        raise ValueError('G0 edge hash mismatch')
    if steps is None:
        steps = config.steps
    if tuple(steps) != config.steps and device != 'cpu':
        raise ValueError('Reduced trajectory is toy CPU only')
    if tuple(steps) == config.steps and (
        provenance['node_count'] != 90_253 or len(state['edges']) != 4_999_981 or
        sum(kind == 'movie' for _,kind,*_ in state['registry']) != 40_666 or
        len(windows) != 12 or windows[0]['start'] != 1577836800 or
        windows[-1]['end'] != 1609459200):
        raise ValueError('Production C0 requires approved 5M G0 and all 2020 windows')
    if steps[0] != 0 or any(a >= b for a,b in zip(steps,steps[1:])) or steps[-1] > config.steps[-1]:
        raise ValueError('Invalid trajectory steps')
    output.mkdir(parents=True,exist_ok=False)
    status_path=output/'report.json'
    status_path.write_text(json.dumps(dict(status='INCOMPLETE',candidates=[]))+'\n')
    torch.manual_seed(config.seed)
    model = LightGCN(provenance['node_count'],config.dim,config.layers).to(device)
    sparse = adjacency.to(device) if isinstance(adjacency,torch.Tensor) else None
    if sparse is None or sparse.layout != torch.sparse_coo:
        raise ValueError('Expected validated sparse G0 tensor')
    optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)
    sampler = sample_batches(state,config.batch_size,steps[-1],config.seed)
    records = []
    for step in range(steps[-1]+1):
        if step:
            bpr_step(model,sparse,optimizer,next(sampler),device)
        if step not in steps:
            continue
        saved = save_candidate(output/f'candidate_{step}.pt',model,step=step,
                               config=config,provenance=provenance,toy_cpu=tuple(steps)!=config.steps)
        model.eval()
        with torch.inference_mode():
            embeddings = model(sparse)
        metrics = aggregate_validation([evaluate_embeddings(embeddings,state['registry'],windows)],len(windows))
        records.append(dict(step=step,validation_recall_20=metrics['validation_recall_20'],
                            validation_ndcg_20=metrics['validation_ndcg_20'],
                            checkpoint_sha256=saved['sha256'],windows=metrics['windows']))
        status_path.write_text(json.dumps(dict(status='INCOMPLETE',candidates=records),indent=2)+'\n')
    if tuple(steps) != config.steps:
        return dict(status='TOY_ONLY',candidates=records,selected=None)
    selected = select_checkpoint(records)
    result=dict(status='COMPLETE',candidates=records,selected_step=selected['step'])
    status_path.write_text(json.dumps(result,indent=2)+'\n')
    return result
