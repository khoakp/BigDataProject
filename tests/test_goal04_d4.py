"""D4-A static and CPU tests; never allocate a CUDA tensor or read real graph rows."""
import json
import tempfile
import unittest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.kaggle_d4_runner import (INPUT_SOURCES, INPUT_VERSIONS, PHASE_CAPS_S,
                                     SELECTION, _phase_resource,
                                     _unique_file, approval_gate, run)
from scripts.prepare_kaggle_d4 import prepare as prepare_package
from scripts.verify_kaggle_d4 import verify
from src.benchmarking.common import PreflightSampler
from src.evaluation.baselines import popularity_counts, popularity_counts_snapshot
from src.evaluation.refresh_strategies import _device, _environment, _gpu_headroom
from src.evaluation.refresh_strategies import ForwardLedger, create_p0, prepare, rank_users
from src.evaluation.refresh_boundary import RankRequest, freeze_boundary
from tests.test_goal04_refresh_toy import toy, ts
from src.graph.model_replay import ModelGraphReplay, load_projection_snapshot


class D4LocalTest(unittest.TestCase):
    def test_cuda_guard_and_environment_are_mock_only(self):
        with self.assertRaisesRegex(ValueError, 'Only cpu'):
            _device('cuda:1')
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'device 0'):
                _device('cuda:0')
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.current_device', return_value=1):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                _device('cuda:0')
        with patch('torch.cuda.mem_get_info', return_value=(29, 100)):
            with self.assertRaises(MemoryError):
                _gpu_headroom()
        with patch('torch.cuda.get_device_properties',
                   return_value=SimpleNamespace(name='Mock T4', total_memory=100)):
            self.assertNotEqual(_environment('cpu'), _environment('cuda:0'))

    def test_process_tree_sampler_and_null_gpu(self):
        with PreflightSampler(interval=.001) as sampler:
            bytes_ = bytearray(1024 * 1024)
            self.assertEqual(len(bytes_), 1024 * 1024)
        record = sampler.record()
        self.assertGreater(record['process_tree_peak_rss_bytes'], 0)
        self.assertGreater(record['min_host_available_ram_bytes'], 0)
        self.assertIsNone(record['gpu_peak_allocated_bytes'])
        self.assertIsNotNone(record['gpu_unavailable_reason'])
        with patch('torch.cuda.mem_get_info', return_value=(60, 100)), \
             patch('torch.cuda.memory_allocated', return_value=12), \
             patch('torch.cuda.memory_reserved', return_value=20):
            gpu = PreflightSampler(device='cuda:0')
            gpu.sample()
        self.assertEqual(gpu.record()['gpu_peak_reserved_bytes'], 20)
        self.assertEqual(gpu.record()['gpu_min_free_fraction'], .6)
        with patch('torch.cuda.mem_get_info', return_value=(60, 100)), \
             patch('torch.cuda.memory_allocated', return_value=12), \
             patch('torch.cuda.memory_reserved', return_value=20), \
             patch('scripts.kaggle_d4_runner.available_ram', return_value=300), \
             patch('scripts.kaggle_d4_runner.shutil.disk_usage',
                   return_value=SimpleNamespace(free=400)):
            phase = _phase_resource(SimpleNamespace(sample=lambda: None,
                                                     peak_rss_bytes=500))
        self.assertEqual(phase['gpu_allocated_bytes_at_end'], 12)
        self.assertEqual(phase['effective_host_available_bytes_at_end'], 300)

    def test_snapshot_counts_match_existing_cpu_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, events, _, _, _, _ = toy(root)
            graph_profile = {}
            with ModelGraphReplay.create(root / 'd4.duckdb', cohort, events,
                    duckdb_config={'memory_limit': '1GB', 'threads': '1'},
                    profile_timings=graph_profile) as graph:
                graph.advance(ts(2021, 1))
                snapshot = root / 'snapshot'
                graph.export_snapshot(snapshot, expected_version=1)
                timing = {}
                load_projection_snapshot(snapshot,
                    expected_mapping=graph.meta['mapping_sha256'], expected_version=1,
                    expected_boundary=ts(2021, 1),
                    expected_source=graph.meta['source_sha256'],
                    materialize=False, profile_timings=timing)
                self.assertIn('snapshot_verify_s', timing)
                state = graph.state()
                movie_ids = {raw: node for node, kind, raw, *_ in state['registry']
                             if kind == 'movie'}
                self.assertEqual(popularity_counts(state, movie_ids),
                                 popularity_counts_snapshot(snapshot, movie_ids))
            self.assertIn('_attach_source_s', graph_profile)
            self.assertIn('_record_s', graph_profile)

    def test_s2_snapshot_path_matches_cpu_strategy_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, events, checkpoint, _, _, _ = toy(root)
            with ModelGraphReplay.create(root / 'd4-s2.duckdb', cohort, events) as graph:
                history = freeze_boundary(graph, ts(2021, 1), root / 'snapshot')
            cpu = prepare('S2', history, checkpoint, repetition=1)
            with patch('src.evaluation.refresh_strategies._device',
                       return_value=torch.device('cpu')):
                streamed = prepare('S2', history, checkpoint, repetition=1,
                                   device='cuda:0')
            request = RankRequest(history, (1, 2), 'serving_like')
            self.assertEqual(rank_users(cpu, request), rank_users(streamed, request))

    def test_mock_device_adapter_matches_cpu_rankings_and_separates_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort, events, checkpoint, _, g0, _ = toy(root)
            with ModelGraphReplay.create(root / 'd4-adapter.duckdb', cohort, events) as graph:
                history = freeze_boundary(graph, ts(2021, 1), root / 'snapshot')
            args = dict(base_boundary=ts(2020, 1),
                        mapping_sha256=checkpoint.provenance['mapping_sha256'],
                        source_sha256=checkpoint.provenance['source_sha256'])
            cpu_id, cpu_sha = create_p0(checkpoint, g0, cache_path=root / 'p0-cpu',
                                        ledger=ForwardLedger(), **args)
            cpu_s1 = prepare('S1', history, checkpoint, repetition=1,
                             cache_path=root / 's1-cpu', ledger=ForwardLedger())
            cpu_s0 = prepare('S0', history, checkpoint, repetition=1,
                             p0_path=root / 'p0-cpu', p0_identity=cpu_id,
                             p0_manifest_sha256=cpu_sha)
            with patch('src.evaluation.refresh_strategies._device',
                       return_value=torch.device('cpu')), \
                 patch('src.evaluation.refresh_strategies._environment',
                       return_value='mock-cuda'):
                mock_id, mock_sha = create_p0(checkpoint, g0,
                    cache_path=root / 'p0-mock', ledger=ForwardLedger(),
                    device='cuda:0', **args)
                mock_s1 = prepare('S1', history, checkpoint, repetition=1,
                    cache_path=root / 's1-mock', ledger=ForwardLedger(), device='cuda:0')
                mock_s0 = prepare('S0', history, checkpoint, repetition=1,
                    p0_path=root / 'p0-mock', p0_identity=mock_id,
                    p0_manifest_sha256=mock_sha, device='cuda:0')
            self.assertNotEqual(cpu_id, mock_id)
            request = RankRequest(history, (1, 2), 'serving_like')
            self.assertEqual(rank_users(cpu_s0, request), rank_users(mock_s0, request))
            self.assertEqual(rank_users(cpu_s1, request), rank_users(mock_s1, request))

    def test_package_is_private_and_approval_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PermissionError):
                approval_gate(root)
            package = root / 'package'
            manifest = prepare_package(package)
            check = verify(package)
            self.assertEqual(check['status'], 'PASS_BLOCKED_PENDING_OWNER')
            self.assertEqual(manifest['status'], 'BLOCKED_OWNER_APPROVAL')
            self.assertEqual(len(manifest['input_sha256']), 4)
            self.assertIn('torch', manifest['build_environment'])
            self.assertFalse(any(name.endswith('.pt') or name.endswith('.parquet')
                                 for name in manifest['source_sha256']))
            altered = dict(manifest)
            altered['input_sha256'] = dict(manifest['input_sha256'], c0='wrong')
            (package / 'package-manifest.json').write_text(json.dumps(altered), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                verify(package)
            (package / 'package-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            metadata_path = package / 'kernel-metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata_path.write_text(json.dumps(dict(metadata, kernel_sources=[])),
                                     encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                verify(package)
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            (package / 'runner.py').write_text('modified', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify(package)

    def test_missing_private_inputs_and_gpu_gate_do_no_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('scripts.kaggle_d4_runner.INPUT', root), \
                 patch('scripts.kaggle_d4_runner.OUTPUT', root / 'output'), \
                 patch('torch.cuda.is_available') as cuda:
                blocked = run()
                self.assertEqual(blocked['status'], 'INCOMPLETE')
                self.assertIsNone(blocked['resources'])
                self.assertIsNotNone(blocked['resources_unavailable_reason'])
                self.assertFalse((root / 'output').exists())
                cuda.assert_not_called()
                with self.assertRaises(ValueError):
                    _unique_file('candidate_1500.pt', 'bad')

    def test_approval_requires_exact_package_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = json.loads(SELECTION.read_text(encoding='utf-8'))
            decision = dict(status='OWNER_APPROVED_D4_B',
                approved_kernel_id='khoakp/moviegraph-v2-d4-resource-preflight',
                approved_source_bundle_sha256='bundle', approved_git_head='head',
                owner_decision_reference='owner-d4b-1', approved_boundary=1609459200,
                approved_device='cuda:0', approved_job_count=1, approved_forward_attempts=2,
                approved_input_sources=INPUT_SOURCES,
                approved_input_versions=INPUT_VERSIONS,
                approved_phase_caps_s=PHASE_CAPS_S,
                approved_input_sha256={
                    'c0': selection['checkpoint']['sha256'],
                    'g0_manifest': selection['g0_snapshot_manifest_sha256'],
                    'canonical': selection['canonical_source_sha256'],
                    'cohort_manifest': selection['cohort_manifest_sha256']},
                p0_accounting='official_setup_1_of_100',
                s1_accounting='official_rep1_boundary1_1_of_99',
                s1_reuse_authorized=True, max_wall_s=1800,
                failed_forward_counts=True, retry_policy='none',
                output_policy='private_fresh_output_then_independent_hash_verification',
                quality_evaluation=False, holdout_access=False,
                min_host_available_bytes=8 * 1024**3,
                min_disk_free_bytes=10 * 1024**3, min_gpu_free_fraction=.30)
            (root / 'd4-owner-approval.json').write_text(json.dumps(decision), encoding='utf-8')
            with self.assertRaises(PermissionError):
                approval_gate(root, bundle_sha256='other', git_head='head')
            self.assertEqual(approval_gate(root, bundle_sha256='bundle', git_head='head'),
                             decision)
            decision['status'] = 'DRAFT_NOT_APPROVED'
            (root / 'd4-owner-approval.json').write_text(json.dumps(decision), encoding='utf-8')
            with self.assertRaises(PermissionError):
                approval_gate(root, bundle_sha256='bundle', git_head='head')
            decision['status'] = 'OWNER_APPROVED_D4_B'
            decision['approved_input_sha256']['c0'] = 'wrong'
            (root / 'd4-owner-approval.json').write_text(json.dumps(decision), encoding='utf-8')
            with self.assertRaises(PermissionError):
                approval_gate(root, bundle_sha256='bundle', git_head='head')
            decision['approved_input_sha256']['c0'] = selection['checkpoint']['sha256']
            decision['approved_forward_attempts'] = 3
            (root / 'd4-owner-approval.json').write_text(json.dumps(decision), encoding='utf-8')
            with self.assertRaises(PermissionError):
                approval_gate(root, bundle_sha256='bundle', git_head='head')


if __name__ == '__main__':
    unittest.main()
