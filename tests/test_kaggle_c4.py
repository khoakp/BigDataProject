"""CPU-only proof of the C4 checkpoint trajectory and production guard."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hashlib
import json

import torch

from scripts.kaggle_c4_runner import train_trajectory, locate_c3_snapshot
from src.graph.equivalence import checksum
from src.models.c0 import C0Config


class KaggleC4Test(unittest.TestCase):
    def test_discovers_exact_c3_snapshot_and_passing_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root/'kernel'/'goal04_c3'/'base_snapshot'
            snapshot.mkdir(parents=True)
            manifest = snapshot/'manifest.json'
            manifest.write_text('{"schema_version":"fixed_cohort_projection_v1"}')
            report = snapshot.parent/'report.json'
            report.write_text(json.dumps({'status':'PASS','gpu_optimizer_steps':0,
                                          'quality_evaluations':0}))
            with (patch('scripts.kaggle_c4_runner.INPUT_ROOT',root),
                  patch('scripts.kaggle_c4_runner.C3_SNAPSHOT_MANIFEST_SHA',
                        hashlib.sha256(manifest.read_bytes()).hexdigest()),
                  patch('scripts.kaggle_c4_runner.C3_REPORT_SHA',
                        hashlib.sha256(report.read_bytes()).hexdigest())):
                found, loaded = locate_c3_snapshot()
                self.assertEqual(found,snapshot)
                self.assertEqual(loaded['status'],'PASS')

    def test_one_toy_trajectory_saves_and_reloads_without_quality(self):
        state = {'registry':[(0,'user',1,0),(1,'movie',10,0),(2,'movie',20,0)],
                 'edges':[(0,1,0)],'degrees':[(0,1),(1,1)]}
        indices = torch.tensor([[0,1],[1,0]])
        sparse = torch.sparse_coo_tensor(indices,torch.ones(2),(3,3)).coalesce()
        provenance = {'mapping_sha256':checksum([r[:3] for r in state['registry']]),
                      'cohort_manifest_sha256':'cohort','snapshot_coo_sha256':'coo',
                      'source_sha256':'source','git_head':'head','base_cutoff_ts':1577836800,
                      'node_count':3,'base_edge_count':1,
                      'base_edges_sha256':checksum(state['edges'])}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'candidates'
            result = train_trajectory(state,sparse,root,config=C0Config(),
                                      provenance=provenance,device='cpu',
                                      steps=(0,1,2),toy_cpu=True)
            self.assertEqual(result['status'],'COMPLETE_CANDIDATES')
            self.assertEqual([r['step'] for r in result['checkpoints']],[0,1,2])
            self.assertEqual(len((root/'steps.jsonl').read_text().splitlines()),2)
            self.assertIsNone(result['selected_step'])
            self.assertEqual(result['validation_quality_evaluations'],0)
            with self.assertRaises(ValueError):
                train_trajectory(state,sparse,Path(temp)/'wrong',config=C0Config(),
                                 provenance=provenance,device='cuda:0')


if __name__ == '__main__':
    unittest.main()
