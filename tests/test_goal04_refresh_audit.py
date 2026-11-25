"""Independent adversarial checks of the D2 CPU toy contracts."""
import json
import math
import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import duckdb

from src.artifacts import file_sha256
from src.benchmarking.goal04_refresh import run_toy
from src.evaluation.refresh_boundary import RankRequest, freeze_boundary, make_quality_bundle, select_serving_users
from src.evaluation.refresh_results import aggregate_split, evaluate_quality, paired_comparisons
from src.evaluation.refresh_strategies import ForwardLedger, _identity, create_p0, prepare, rank_users
from src.graph.model_replay import ModelGraphReplay
from src.models.c0 import save_candidate
from src.models.refresh_cache import cache_key, load_verified, parameter_sha256
from src.models.lightgcn import LightGCN
from tests.test_goal04_refresh_toy import toy, ts


class RefreshAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cohort, self.source, self.checkpoint, self.events, self.g0, self.cuts = toy(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def graph(self, name):
        return ModelGraphReplay.create(self.root / f'{name}.duckdb', self.cohort, self.source)

    def p0(self):
        return create_p0(self.checkpoint, self.g0, base_boundary=ts(2020, 1),
                         mapping_sha256=self.checkpoint.provenance['mapping_sha256'],
                         source_sha256=self.checkpoint.provenance['source_sha256'],
                         cache_path=self.root / 'p0', ledger=ForwardLedger())

    def test_truncated_ranking_rejected(self):
        with self.graph('rank') as graph:
            history = freeze_boundary(graph, self.cuts[0], self.root / 'rank-snapshot')
            bundle = make_quality_bundle(graph, history, self.cuts[1])
            ranks = {s: {u: [] for u in bundle.eligible_user_ids} for s in ('S0', 'S1', 'S2')}
            with self.assertRaises(ValueError):
                evaluate_quality(bundle, ranks)

    def test_paired_and_macro_mismatched_rows_rejected(self):
        with self.graph('paired') as graph:
            history = freeze_boundary(graph, self.cuts[0], self.root / 'paired-snapshot')
            bundle = make_quality_bundle(graph, history, self.cuts[1])
            ranks = {s: {u: [(m, float(-m)) for m, _ in history.movie_ids
                              if m not in history.mask.seen_for((u,))[u]]
                         for u in bundle.eligible_user_ids} for s in ('S0', 'S1', 'S2')}
            result = evaluate_quality(bundle, ranks)
            rows = deepcopy(result['per_user'])
            rows[1]['relevant_count'] += 1
            with self.assertRaises(ValueError):
                paired_comparisons(rows)
            windows = deepcopy(result['windows'])
            windows[1]['eligible_users'] += 1
            with self.assertRaises(ValueError):
                aggregate_split(windows, split='toy')

    def test_hand_calculated_quality_and_popularity(self):
        with self.graph('oracle') as graph:
            history = freeze_boundary(graph, ts(2020, 4), self.root / 'oracle-snapshot')
            bundle = make_quality_bundle(graph, history, ts(2020, 5))
            self.assertEqual(bundle.eligible_label_pairs, ((1, 20),))
            ranks = {s: {1: [(30, 2.0), (20, 1.0)]} for s in ('S0', 'S1', 'S2')}
            result = evaluate_quality(bundle, ranks)
            self.assertAlmostEqual(result['windows'][0]['mean_recall_20'], 1.0)
            self.assertAlmostEqual(result['windows'][0]['mean_ndcg_20'],
                                   1 / math.log2(3))
            history = freeze_boundary(graph, self.cuts[0], self.root / 'oracle-jan-snapshot')
            state = prepare('S2', history, self.checkpoint, repetition=1)
            self.assertEqual(state.scores[1][10], 1)
            self.assertEqual(state.scores[1][20], 2)
            self.assertEqual(state.scores[1][30], 1)

    def test_cache_manifest_requires_external_anchor(self):
        identity, sha = self.p0()
        path = self.root / 'p0'
        tensor, manifest = load_verified(path, identity, expected_manifest_sha256=sha)
        self.assertEqual(tuple(tensor.shape), (5, 64))
        # Attacker rewrites both files consistently under an unchanged identity.
        torch.save(torch.zeros_like(tensor), path / 'embeddings.pt')
        manifest['embeddings_sha256'] = file_sha256(path / 'embeddings.pt')
        (path / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True,
                                            separators=(',', ':')) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'external manifest SHA-256 required'):
            load_verified(path, identity)

    def test_cache_malformed_metadata_tensor_and_scope(self):
        identity, sha = self.p0()
        original = self.root / 'p0'
        for kind in ('truncated', 'key', 'dtype', 'inf', 'wrong_env'):
            path = self.root / kind
            shutil.copytree(original, path)
            manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
            if kind == 'truncated':
                (path / 'manifest.json').write_bytes(b'{"schema_version":')
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    load_verified(path, identity, expected_manifest_sha256=file_sha256(path / 'manifest.json'))
                continue
            if kind == 'key':
                manifest['cache_key'] = '0' * 64
            if kind == 'wrong_env':
                wrong = dict(identity, environment='wrong-environment')
                with self.assertRaises(ValueError):
                    load_verified(path, wrong, expected_manifest_sha256=sha)
                continue
            if kind in ('dtype', 'inf'):
                tensor, _ = load_verified(path, identity, expected_manifest_sha256=sha)
                tensor = tensor.double() if kind == 'dtype' else torch.full_like(tensor, float('inf'))
                torch.save(tensor, path / 'embeddings.pt')
                manifest['embeddings_sha256'] = file_sha256(path / 'embeddings.pt')
                manifest['dtype'] = str(tensor.dtype)
            (path / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True,
                                                separators=(',', ':')) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_verified(path, identity, expected_manifest_sha256=file_sha256(path / 'manifest.json'))

    def test_mutating_forward_detected_and_counted(self):
        with self.graph('mutating') as graph:
            history = freeze_boundary(graph, self.cuts[0], self.root / 'mutating-snapshot')
            ledger = ForwardLedger()
            original = LightGCN.forward

            def mutate(model, adjacency):
                with torch.no_grad():
                    model.embedding.weight[0, 0] += 1
                return original(model, adjacency)

            with patch.object(LightGCN, 'forward', mutate), self.assertRaisesRegex(ValueError, 'parameters changed'):
                prepare('S1', history, self.checkpoint, repetition=1,
                        cache_path=self.root / 'mutating-cache', ledger=ledger)
            self.assertEqual(ledger.attempts, 1)
            self.assertEqual(ledger.optimizer_updates, 0)
            self.assertFalse((self.root / 'mutating-cache').exists())

    def test_boundary_ties_low_rating_and_label_capability(self):
        tied = [(eid, u, m, src) for eid, stamp, u, m, _, src in self.events
                if stamp == self.cuts[0]]
        self.assertEqual(len(tied), 2)
        self.assertEqual(tied, sorted(tied, key=lambda x: (x[1], x[2], x[3])))
        self.assertEqual([row[0] for row in tied], list(range(tied[0][0], tied[0][0] + 2)))
        feb_tied = [(eid, u, m, src) for eid, stamp, u, m, _, src in self.events
                    if stamp == self.cuts[1]]
        self.assertEqual(len(feb_tied), 2)
        self.assertEqual(feb_tied, sorted(feb_tied, key=lambda x: (x[1], x[2], x[3])))
        with self.graph('ties') as graph:
            history = freeze_boundary(graph, self.cuts[0], self.root / 'ties-snapshot')
            self.assertEqual(history.mask.seen_for((1, 2)), {1: {10, 20}, 2: {20, 30}})
            self.assertEqual(select_serving_users(graph, history).selected, (1,))
            self.assertFalse(hasattr(history, 'eligible_label_pairs'))
            self.assertFalse(hasattr(RankRequest(history, (1,), 'serving_like'), 'eligible_label_pairs'))
            bundle = make_quality_bundle(graph, history, self.cuts[1])
            self.assertEqual(bundle.eligible_label_pairs, ((1, 30), (2, 10)))
            denied = ForwardLedger()
            with self.assertRaises(TypeError):
                prepare('S1', bundle, self.checkpoint, repetition=1,
                        cache_path=self.root / 'label-bearing-cache', ledger=denied)
            self.assertEqual(denied.attempts, 0)
            self.assertTrue(any(stamp == self.cuts[1] and rating >= 4
                                for _, stamp, _, _, rating, _ in self.events))
            self.assertEqual(bundle.positive_events, sum(
                self.cuts[0] <= stamp < self.cuts[1] and rating >= 4
                for _, stamp, _, _, rating, _ in self.events))
            self.assertEqual(bundle.within_window_repeat_events, 1)
            self.assertLess(history.last_canonical_order_key[0], self.cuts[0])
            after = freeze_boundary(graph, self.cuts[1], self.root / 'ties-next-snapshot')
            self.assertEqual(after.mask.seen_for((1, 2)), {1: {10, 20, 30}, 2: {10, 20, 30}})

    def test_canonical_mutation_after_freeze_rejected(self):
        with self.graph('source') as graph:
            history = freeze_boundary(graph, self.cuts[0], self.root / 'source-snapshot')
            changed = self.root / 'changed.parquet'
            with duckdb.connect() as con:
                con.execute('CREATE TABLE changed AS SELECT * FROM read_parquet(?) ORDER BY event_id DESC',
                            [str(self.source)])
                con.execute('COPY changed TO ? (FORMAT PARQUET)', [str(changed)])
            changed.replace(self.source)
            with self.assertRaises(ValueError):
                select_serving_users(graph, history)
            with self.assertRaises(ValueError):
                make_quality_bundle(graph, history, self.cuts[1])

    def test_future_only_source_change_alters_cache_identity(self):
        changed = self.root / 'future-only.parquet'
        with duckdb.connect() as con:
            con.execute('''CREATE TABLE altered AS SELECT event_id,event_ts,user_raw_id,
                movie_raw_id, CASE WHEN event_ts>=? AND user_raw_id=1 AND movie_raw_id=30
                THEN 3.0 ELSE rating END AS rating,source_row_id FROM read_parquet(?)''',
                [self.cuts[0], str(self.source)])
            con.execute('COPY altered TO ? (FORMAT PARQUET)', [str(changed)])
        with self.graph('original-key') as original, ModelGraphReplay.create(
                self.root / 'altered-key.duckdb', self.cohort, changed) as altered:
            altered_g0 = self.root / 'altered-g0'
            altered.export_snapshot(altered_g0, expected_version=0)
            altered_provenance = dict(self.checkpoint.provenance,
                                      source_sha256=altered.meta['source_sha256'])
            altered_checkpoint_path = self.root / 'altered-c0.pt'
            saved = save_candidate(altered_checkpoint_path, self.checkpoint.load(),
                                   step=0, config=self.checkpoint.config,
                                   provenance=altered_provenance, toy_cpu=True)
            altered_checkpoint = replace(
                self.checkpoint, path=altered_checkpoint_path, sha256=saved['sha256'],
                provenance=altered_provenance,
                g0_manifest_sha256=file_sha256(altered_g0 / 'manifest.json'))
            self.assertEqual(parameter_sha256(self.checkpoint.load()),
                             parameter_sha256(altered_checkpoint.load()))
            before = freeze_boundary(original, self.cuts[0], self.root / 'original-key-snapshot')
            after = freeze_boundary(altered, self.cuts[0], self.root / 'altered-key-snapshot')
            self.assertEqual(original.state()['edges'], altered.state()['edges'])
            self.assertEqual(original.state()['degrees'], altered.state()['degrees'])
            self.assertEqual(before.mapping_sha256, after.mapping_sha256)
            self.assertEqual(before.coo_sha256, after.coo_sha256)
            self.assertEqual(before.mask.seen_for((1, 2)), after.mask.seen_for((1, 2)))
            self.assertNotEqual(before.source_sha256, after.source_sha256)
            manifest = json.loads((before.snapshot / 'manifest.json').read_text(encoding='utf-8'))
            identity = _identity('S1', self.checkpoint, manifest, before.snapshot_sha256, 1)
            self.assertNotEqual(cache_key(identity), cache_key(dict(
                identity, source_sha256=after.source_sha256)))
            self.assertEqual(select_serving_users(original, before).selected,
                             select_serving_users(altered, after).selected)
            cross_source = ForwardLedger()
            with self.assertRaises(ValueError):
                prepare('S1', after, self.checkpoint, repetition=1,
                        cache_path=self.root / 'cross-source-s1', ledger=cross_source)
            self.assertEqual(cross_source.attempts, 0)
            original_p0_identity, original_p0_sha = self.p0()
            altered_p0_path = self.root / 'altered-p0'
            altered_p0_identity, altered_p0_sha = create_p0(
                altered_checkpoint, altered_g0, base_boundary=ts(2020, 1),
                mapping_sha256=after.mapping_sha256, source_sha256=after.source_sha256,
                cache_path=altered_p0_path, ledger=ForwardLedger())
            original_states = {
                'S0': prepare('S0', before, self.checkpoint, repetition=1,
                              p0_path=self.root / 'p0', p0_identity=original_p0_identity,
                              p0_manifest_sha256=original_p0_sha),
                'S1': prepare('S1', before, self.checkpoint, repetition=1,
                              cache_path=self.root / 'original-s1', ledger=ForwardLedger()),
                'S2': prepare('S2', before, self.checkpoint, repetition=1),
            }
            altered_states = {
                'S0': prepare('S0', after, altered_checkpoint, repetition=1,
                              p0_path=altered_p0_path, p0_identity=altered_p0_identity,
                              p0_manifest_sha256=altered_p0_sha),
                'S1': prepare('S1', after, altered_checkpoint, repetition=1,
                              cache_path=self.root / 'altered-s1', ledger=ForwardLedger()),
                'S2': prepare('S2', after, altered_checkpoint, repetition=1),
            }
            for strategy in ('S0', 'S1', 'S2'):
                if strategy != 'S2':
                    self.assertTrue(torch.equal(original_states[strategy].scores,
                                                altered_states[strategy].scores))
                self.assertEqual(rank_users(original_states[strategy], RankRequest(
                    before, (1, 2), 'serving_like')),
                    rank_users(altered_states[strategy], RankRequest(
                        after, (1, 2), 'serving_like')))

    def test_strategy_costs_and_failure_location(self):
        actual_order = []

        def tracked_prepare(strategy, *args, **kwargs):
            actual_order.append((kwargs['repetition'], strategy))
            return prepare(strategy, *args, **kwargs)

        with (patch('torch.optim.Adam', side_effect=AssertionError('optimizer forbidden')),
              patch('src.benchmarking.goal04_refresh.prepare', side_effect=tracked_prepare)):
            report = run_toy(self.cohort, self.source, self.checkpoint, self.g0,
                             ((self.cuts[0], self.cuts[1]),), self.root / 'run')
        self.assertEqual(report['status'], 'COMPLETE_TOY_CPU')
        self.assertEqual(actual_order, [(1, 'S0'), (1, 'S1'), (1, 'S2'),
                                        (2, 'S1'), (2, 'S2'), (2, 'S0'),
                                        (3, 'S2'), (3, 'S0'), (3, 'S1')])
        self.assertEqual(len({r['windows'][0]['catalog_sha256'] for r in report['repetitions']}), 1)
        self.assertEqual(len({r['windows'][0]['mask_sha256'] for r in report['repetitions']}), 1)
        row = report['repetitions'][0]['windows'][0]
        self.assertGreaterEqual(report['repetitions'][0]['replay_initialization_s'], 0)
        self.assertEqual(set(row['strategy_timings']), {'S0', 'S1', 'S2'})
        self.assertEqual(row['cache_disposition'], {'S0': 'HIT', 'S1': 'MISS_PUBLISHED', 'S2': 'NOT_APPLICABLE'})
        self.assertIn('quality_bundle_s', row['timings'])
        for key in ('serving_like_rank_s', 'offline_rank_s', 'cache_io_s', 'propagation_s'):
            self.assertAlmostEqual(row['timings'][key], sum(
                scoped.get(key, 0) for scoped in row['strategy_timings'].values()))
        with patch('src.benchmarking.goal04_refresh.prepare', side_effect=TimeoutError('injected timeout')):
            failed = run_toy(self.cohort, self.source, self.checkpoint, self.g0,
                             ((self.cuts[0], self.cuts[1]),), self.root / 'failed')
        self.assertEqual(failed['status'], 'INCOMPLETE')
        self.assertIn('TimeoutError', failed['error'])
        self.assertEqual(failed['failure']['rep'], 1)
        self.assertEqual(failed['failure']['strategy'], 'S0')
        self.assertEqual(failed['failure']['b'], self.cuts[0])

    def test_failed_s1_forward_marks_repetition_incomplete(self):
        def fail_s1(strategy, *args, **kwargs):
            if strategy == 'S1':
                with patch.object(LightGCN, 'forward', side_effect=RuntimeError('injected forward')):
                    return prepare(strategy, *args, **kwargs)
            return prepare(strategy, *args, **kwargs)

        with patch('src.benchmarking.goal04_refresh.prepare', side_effect=fail_s1):
            report = run_toy(self.cohort, self.source, self.checkpoint, self.g0,
                             ((self.cuts[0], self.cuts[1]),), self.root / 'failed-s1')
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['failure']['strategy'], 'S1')
        self.assertEqual(report['failure']['stage'], 'strategy_prepare')
        self.assertEqual(report['p0_setup_forward_attempts'], 1)
        self.assertEqual(report['s1_forward_attempts'], 1)
        self.assertEqual(report['repetitions'][0]['status'], 'INCOMPLETE')
        self.assertEqual(report['repetitions'][0]['forward_attempts'], 1)


if __name__ == '__main__':
    unittest.main()
