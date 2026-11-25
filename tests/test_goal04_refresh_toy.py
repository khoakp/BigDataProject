"""D2-only synthetic CPU gates; no official C0/G0 or future quality data."""
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import duckdb
import torch

from src.artifacts import file_sha256
from src.benchmarking.goal04_refresh import run_toy
from src.evaluation.c0_validation import score_batched
from src.evaluation.refresh_boundary import (RankRequest, freeze_boundary,
                                             make_quality_bundle, select_serving_users)
from src.evaluation.refresh_results import aggregate_split, evaluate_quality, paired_comparisons
from src.evaluation.refresh_strategies import (ForwardLedger, VerifiedC0, counters,
                                               create_p0, get_scores, prepare, provenance, rank_users)
from src.graph.equivalence import checksum
from src.graph.model_replay import ModelGraphReplay, load_projection_snapshot
from src.models.c0 import C0Config, save_candidate
from src.models.lightgcn import LightGCN
from src.models.refresh_cache import load_verified, parameter_sha256, publish_atomic
from tests.test_model_replay import reference as full_reference


def ts(year, month, day=1):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def toy(root):
    """Base nodes 2 users/3 movies; canonical also contains omitted/cold nodes."""
    base = ts(2020, 1)
    jan, feb, mar = ts(2021, 1), ts(2021, 2), ts(2021, 3)
    raw = [
        (ts(2019, 12, 1), 1, 10, 5.), (ts(2019, 12, 2), 2, 20, 4.),
        (ts(2019, 12, 3), 2, 30, 4.), (ts(2019, 12, 4), 3, 40, 5.),
        (ts(2020, 4), 1, 20, 5.), (ts(2020, 12, 20), 1, 20, 5.),
        (ts(2020, 6, 21), 2, 20, 4.),
        (ts(2020, 12, 22), 2, 10, 2.),
        (jan, 1, 30, 5.), (jan, 1, 30, 4.), (jan + 1, 2, 10, 4.),
        (jan + 2, 4, 10, 5.), (jan + 3, 3, 10, 5.),
        (jan + 4, 1, 40, 5.), (jan + 5, 1, 50, 5.),
        (jan + 6, 1, 10, 2.), (feb, 1, 20, 1.),
        (feb, 2, 10, 4.), (feb + 2, 1, 30, 5.),
        (mar, 1, 20, 5.),
    ]
    ordered = sorted(((stamp, user, movie, rating, src)
                      for src, (stamp, user, movie, rating) in enumerate(raw)),
                     key=lambda row: (row[0], row[1], row[2], row[4]))
    events = [(eid, stamp, user, movie, rating, src)
              for eid, (stamp, user, movie, rating, src) in enumerate(ordered)]
    cohort = root / 'cohort'
    cohort.mkdir()
    source = root / 'canonical.parquet'
    nodes = [(0, 'user', 1), (1, 'user', 2), (2, 'movie', 10),
             (3, 'movie', 20), (4, 'movie', 30)]
    edges = [(0, 2, ts(2019, 12, 1)), (1, 3, ts(2019, 12, 2)),
             (1, 4, ts(2019, 12, 3))]
    with duckdb.connect() as con:
        con.execute('CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating DOUBLE,source_row_id BIGINT)')
        con.executemany('INSERT INTO e VALUES (?,?,?,?,?,?)', events)
        con.execute('COPY e TO ? (FORMAT PARQUET)', [str(source)])
        con.execute('CREATE TABLE n(node_id BIGINT,node_type VARCHAR,raw_id BIGINT)')
        con.executemany('INSERT INTO n VALUES (?,?,?)', nodes)
        con.execute('COPY n TO ? (FORMAT PARQUET)', [str(cohort / 'nodes.parquet')])
        con.execute('CREATE TABLE g(user_node_id BIGINT,movie_node_id BIGINT,first_event_ts BIGINT)')
        con.executemany('INSERT INTO g VALUES (?,?,?)', edges)
        con.execute('COPY g TO ? (FORMAT PARQUET)', [str(cohort / 'edges.parquet')])
    (cohort / 'manifest.json').write_text(json.dumps(dict(
        cutoff_ts=base, positive_threshold=4.0, positive_edges=3,
        graph_semantics='complete_induced_pre_cutoff_positive',
        nodes_sha256=file_sha256(cohort / 'nodes.parquet'),
        edges_sha256=file_sha256(cohort / 'edges.parquet'))), encoding='utf-8')
    with ModelGraphReplay.create(root / 'initial.duckdb', cohort, source) as graph:
        g0 = root / 'initial-g0'
        m = graph.export_snapshot(g0, expected_version=0)
        state = graph.state()
        provenance = dict(mapping_sha256=graph.meta['mapping_sha256'],
                          cohort_manifest_sha256=graph.meta['cohort_manifest_sha256'],
                          snapshot_coo_sha256=m['coo_sha256'],
                          source_sha256=graph.meta['source_sha256'], git_head='synthetic',
                          base_cutoff_ts=base, node_count=5, base_edge_count=3,
                          base_edges_sha256=checksum(state['edges']))
    torch.manual_seed(7)
    model = LightGCN(5, 64, 3)
    checkpoint_path = root / 'synthetic-c0.pt'
    saved = save_candidate(checkpoint_path, model, step=0, config=C0Config(),
                           provenance=provenance, toy_cpu=True)
    checkpoint = VerifiedC0(checkpoint_path, saved['sha256'], provenance,
                            file_sha256(g0 / 'manifest.json'), toy_cpu=True)
    return cohort, source, checkpoint, events, g0, (jan, feb, mar)


class RefreshToyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cohort, self.source, self.checkpoint, self.events, self.g0, self.cuts = toy(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _graph(self, name):
        return ModelGraphReplay.create(self.root / f'{name}.duckdb', self.cohort, self.source)

    def _p0(self):
        ledger = ForwardLedger()
        identity, sha = create_p0(self.checkpoint, self.g0,
                                  base_boundary=ts(2020, 1),
                                  mapping_sha256=self.checkpoint.provenance['mapping_sha256'],
                                  source_sha256=self.checkpoint.provenance['source_sha256'],
                                  cache_path=self.root / 'p0', ledger=ledger)
        self.assertEqual(ledger.attempts, 1)
        return identity, sha

    def test_boundary_ties_mask_exclusions_and_serving_no_future(self):
        jan, feb, _ = self.cuts
        with self._graph('boundary') as graph:
            history = freeze_boundary(graph, jan, self.root / 'snap-jan')
            self.assertLess(history.last_canonical_order_key[0], jan)
            self.assertEqual(history.mask.seen_for((1, 2)), {1: {10, 20}, 2: {20, 30}})
            self.assertEqual(select_serving_users(graph, history).selected, (1,))
            self.assertNotIn(10, history.mask.seen_for((2,))[2])
            self.assertFalse(hasattr(history, 'eligible_label_pairs'))
            bundle = make_quality_bundle(graph, history, feb)
            self.assertEqual(bundle.eligible_label_pairs, ((1, 30), (2, 10)))
            self.assertEqual(dict(bundle.exclusions), dict(cold_user=1, omitted_user=1,
                                                            cold_movie=1, omitted_movie=1,
                                                            seen=0, eligible=2))
            self.assertEqual(bundle.within_window_repeat_events, 1)
            self.assertEqual(bundle.unique_positive_pairs, sum(dict(bundle.exclusions).values()))
            self.assertEqual(history.mask.seen_for((1,))[1], {10, 20})
            next_history = freeze_boundary(graph, feb, self.root / 'snap-feb')
            self.assertEqual(next_history.mask.seen_for((1, 2)),
                             {1: {10, 20, 30}, 2: {10, 20, 30}})
            self.assertEqual(select_serving_users(graph, next_history).status, 'NOT_APPLICABLE')
            empty = make_quality_bundle(graph, next_history, self.cuts[2])
            self.assertFalse(empty.eligible_label_pairs)
            self.assertGreater(dict(empty.exclusions)['seen'], 0)
            self.assertEqual(empty.no_candidate_users, 2)

    def test_projection_equivalence_normalization_idempotence_rollback(self):
        # Independent full reference for the actual toy fixture.
        jan, feb, _ = self.cuts
        mapping = [(0, 'user', 1), (1, 'user', 2), (2, 'movie', 10),
                   (3, 'movie', 20), (4, 'movie', 30)]
        with self._graph('equivalence') as graph:
            for b in (jan, feb):
                before = graph.current()['projection_version']
                history = freeze_boundary(graph, b, self.root / f'equiv-{b}')
                edges, degrees, coo = full_reference(self.events, mapping, b)
                self.assertEqual(graph.state()['edges'], edges)
                self.assertEqual(graph.state()['degrees'], degrees)
                m, state, adjacency = load_projection_snapshot(
                    history.snapshot, expected_mapping=history.mapping_sha256,
                    expected_version=history.projection_version,
                    expected_boundary=b, expected_source=history.source_sha256)
                self.assertEqual(state['edges'], edges)
                self.assertEqual(len(adjacency.entries), len(coo))
                for actual, expected in zip(adjacency.entries, coo):
                    self.assertEqual(actual[:2], expected[:2])
                    self.assertAlmostEqual(actual[2], expected[2], places=12)
                self.assertEqual(graph.advance(b)['status'], 'NO_OP')
                self.assertEqual(graph.current()['projection_version'], before + 1)
            before = graph.state()
            original = graph._record
            def fail(*args):
                original(*args)
                raise RuntimeError('injected transaction failure')
            with patch.object(graph, '_record', side_effect=fail):
                with self.assertRaises(RuntimeError):
                    graph.advance(self.cuts[2])
            self.assertEqual(graph.state(), before)

    def test_p0_s1_s2_fairness_counts_and_ties(self):
        identity, sha = self._p0()
        jan = self.cuts[0]
        with self._graph('strategies') as graph:
            history = freeze_boundary(graph, jan, self.root / 'strategy-snapshot')
            ledger = ForwardLedger()
            states = {s: prepare(s, history, self.checkpoint, repetition=1,
                                 cache_path=self.root / 's1' if s == 'S1' else None,
                                 p0_path=self.root / 'p0', p0_identity=identity,
                                 p0_manifest_sha256=sha, ledger=ledger)
                      for s in ('S0', 'S1', 'S2')}
            self.assertEqual(ledger.attempts, 1)
            self.assertEqual([counters(states[s])['propagation_forwards'] for s in states], [0, 1, 0])
            self.assertTrue(all(counters(s)['optimizer_updates'] == 0 for s in states.values()))
            self.assertEqual(provenance(states['S0'])['mask_sha256'],
                             provenance(states['S2'])['mask_sha256'])
            request = RankRequest(history, (1, 2), 'serving_like')
            ranks = {s: rank_users(states[s], request) for s in states}
            self.assertEqual(set(map(tuple, (ranks[s].keys() for s in states))), {(1, 2)})
            self.assertEqual([m for m, _ in ranks['S2'][1]], [30])
            self.assertEqual([m for m, _ in ranks['S2'][2]], [10])
            self.assertEqual(ranks['S2'][1][0][1], 1.0)  # future Jan positive not counted
            self.assertEqual(len(states['S2'].scores[0]['edges']), 4)  # repeat is one edge
            self.assertEqual(get_scores(states['S2'])[20], 2)
            detached = get_scores(states['S0'])
            detached.zero_()
            self.assertFalse(torch.equal(detached, states['S0'].scores))
            self.assertEqual(parameter_sha256(self.checkpoint.load()),
                             load_verified(self.root / 's1', states['S1'].identity,
                                           expected_manifest_sha256=states['S1'].cache_manifest_sha256)[1]['parameter_after'])
            bundle = make_quality_bundle(graph, history, self.cuts[1])
            offline = {s: rank_users(states[s], RankRequest(history, bundle.eligible_user_ids,
                                                             'offline_quality')) for s in states}
            quality = evaluate_quality(bundle, offline)
            self.assertEqual(len(quality['per_user']), 6)
            self.assertEqual(len({row['bundle_sha256'] for row in quality['per_user']}), 1)
            self.assertEqual(len(paired_comparisons(quality['per_user'])['windows']), 1)
            self.assertEqual(aggregate_split(quality['windows'], split='toy')['strategies']['S0']['windows'], 1)
            again = prepare('S0', history, self.checkpoint, repetition=2,
                            p0_path=self.root / 'p0', p0_identity=identity,
                            p0_manifest_sha256=sha)
            self.assertEqual(counters(again)['propagation_forwards'], 0)
            self.assertTrue(torch.equal(states['S0'].scores, again.scores))
            with self.assertRaises(ValueError):
                rank_users(states['S0'], RankRequest(replace(history, catalog_sha256='bad'),
                                                     (1,), 'serving_like'))
            with self.assertRaises(ValueError):
                prepare('S2', replace(history, movie_ids=((10, 2),)), self.checkpoint,
                        repetition=1)
            with self.assertRaises(ValueError):
                RankRequest(history, (4,), 'serving_like')
            with self.assertRaises(ValueError):
                RankRequest(history, (1,), 'serving_like', k=10)
        embeddings = torch.zeros((22, 2))
        tie = score_batched(embeddings, {1: 0}, {movie: movie for movie in range(1, 22)}, {}, k=20)
        self.assertEqual([m for m, _ in tie[1]], list(range(1, 21)))

    def test_cache_tamper_nonfinite_and_atomic_failure(self):
        identity, sha = self._p0()
        path = self.root / 'p0'
        tensor, manifest = load_verified(path, identity, expected_manifest_sha256=sha)
        for key, value in [('checkpoint_sha256', 'wrong'), ('snapshot_sha256', 'wrong'),
                           ('mapping_sha256', 'wrong'), ('boundary', identity['boundary'] + 1),
                           ('projection_version', 3), ('device', 'cuda:0'), ('repetition', 1)]:
            wrong = dict(identity, **{key: value})
            with self.assertRaises(ValueError):
                load_verified(path, wrong, expected_manifest_sha256=sha)
        with self.assertRaises(FileNotFoundError):
            load_verified(self.root / 'missing', identity, expected_manifest_sha256=sha)
        original = (path / 'embeddings.pt').read_bytes()
        (path / 'embeddings.pt').write_bytes(original + b'corrupt')
        with self.assertRaises(ValueError):
            load_verified(path, identity, expected_manifest_sha256=sha)
        (path / 'embeddings.pt').write_bytes(original)
        torch.save(tensor[:1], path / 'embeddings.pt')
        altered = dict(manifest, embeddings_sha256=file_sha256(path / 'embeddings.pt'), shape=[1, 64])
        (path / 'manifest.json').write_text(json.dumps(altered, sort_keys=True,
            separators=(',', ':')) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            load_verified(path, identity, expected_manifest_sha256=sha)
        (path / 'embeddings.pt').write_bytes(original)
        (path / 'manifest.json').write_bytes((json.dumps(manifest, sort_keys=True,
            separators=(',', ':')) + '\n').encode())
        raw = (path / 'manifest.json').read_bytes()
        (path / 'manifest.json').write_bytes(raw + b' ')
        with self.assertRaises(ValueError):
            load_verified(path, identity, expected_manifest_sha256=sha)
        (path / 'manifest.json').write_bytes(raw)
        bad = torch.full_like(tensor, float('nan'))
        with self.assertRaises(ValueError):
            publish_atomic(self.root / 'nan', bad, identity, 'same', 'same')
        with self.assertRaises(ValueError):
            publish_atomic(self.root / 'shape', tensor[:1], identity, 'same', 'same')
        with self.assertRaises(ValueError):
            publish_atomic(self.root / 'parameters', tensor, identity, 'before', 'after')
        with patch('src.models.refresh_cache.torch.save', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):
                publish_atomic(self.root / 'interrupted', tensor, identity, 'same', 'same')
        self.assertFalse((self.root / 'interrupted').exists())
        self.assertFalse(list(self.root.glob('interrupted.partial-*')))
        partial = self.root / 'partial'
        partial.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_verified(partial, identity, expected_manifest_sha256=sha)

    def test_pre_and_post_forward_failures_and_stale_provenance(self):
        jan = self.cuts[0]
        p0_ledger = ForwardLedger()
        with self.assertRaises(ValueError):
            create_p0(replace(self.checkpoint, g0_manifest_sha256='wrong'), self.g0,
                      base_boundary=ts(2020, 1),
                      mapping_sha256=self.checkpoint.provenance['mapping_sha256'],
                      source_sha256=self.checkpoint.provenance['source_sha256'],
                      cache_path=self.root / 'bad-g0', ledger=p0_ledger)
        self.assertEqual(p0_ledger.attempts, 0)
        with self._graph('failures') as graph:
            history = freeze_boundary(graph, jan, self.root / 'failure-snapshot')
            ledger = ForwardLedger()
            with self.assertRaises(ValueError):
                prepare('S1', replace(history, projection_version=99), self.checkpoint,
                        repetition=1, cache_path=self.root / 'invalid', ledger=ledger)
            self.assertEqual(ledger.attempts, 0)
            wrong = replace(self.checkpoint, sha256='wrong')
            with self.assertRaises(ValueError):
                prepare('S1', history, wrong, repetition=1,
                        cache_path=self.root / 'wrong-checkpoint', ledger=ledger)
            self.assertEqual(ledger.attempts, 0)
            model = self.checkpoint.load()
            with patch.object(VerifiedC0, 'load', return_value=model), patch.object(
                    model, 'forward', side_effect=RuntimeError('forward failed')):
                with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                    prepare('S1', history, self.checkpoint, repetition=1,
                            cache_path=self.root / 'forward-failed', ledger=ledger)
            self.assertEqual(ledger.attempts, 1)
            self.assertFalse((self.root / 'forward-failed').exists())
            with patch.object(VerifiedC0, 'load', return_value=model), patch.object(
                    model, 'forward', return_value=torch.full((5, 64), float('nan'))):
                with self.assertRaisesRegex(ValueError, 'Invalid propagated'):
                    prepare('S1', history, self.checkpoint, repetition=1,
                            cache_path=self.root / 'nonfinite', ledger=ledger)
            self.assertEqual(ledger.attempts, 2)
            self.assertFalse((self.root / 'nonfinite').exists())

    def test_three_repetitions_and_empty_window(self):
        jan, feb, mar = self.cuts
        report = run_toy(self.cohort, self.source, self.checkpoint, self.g0,
                         ((jan, feb), (feb, mar)), self.root / 'run')
        self.assertEqual(report['status'], 'COMPLETE_TOY_CPU', report['error'])
        self.assertEqual(report['p0_setup_forward_attempts'], 1)
        self.assertEqual(report['s1_forward_attempts'], 6)
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertEqual([tuple(r['strategy_order']) for r in report['repetitions']],
                         [('S0', 'S1', 'S2'), ('S1', 'S2', 'S0'), ('S2', 'S0', 'S1')])
        self.assertEqual([len(r['quality_windows']) for r in report['repetitions']], [6, 0, 0])
        self.assertEqual(report['repetitions'][0]['quality_windows'][3]['mean_recall_20'], None)
        self.assertTrue(all(r['forward_attempts'] == 2 for r in report['repetitions']))
        self.assertEqual(report['repetitions'][0]['windows'][1]['benchmark_status'], 'NOT_APPLICABLE')
        for rep in report['repetitions']:
            for window in rep['windows']:
                self.assertGreaterEqual(window['timings']['serving_like_rank_s'], 0)
                self.assertEqual(window['timings']['offline_rank_s'] == 0, rep['rep'] != 1)
        self.assertTrue((self.root / 'run' / 'report.json').is_file())

    def test_coordinator_failure_status_and_holdout_gate(self):
        jan, feb, _ = self.cuts
        with self.assertRaises(ValueError):
            run_toy(self.cohort, self.source, self.checkpoint, self.g0,
                    ((ts(2023, 1), ts(2023, 2)),), self.root / 'forbidden')
        self.assertFalse((self.root / 'forbidden').exists())
        bad = replace(self.checkpoint, sha256='wrong')
        report = run_toy(self.cohort, self.source, bad, self.g0,
                         ((jan, feb),), self.root / 'incomplete')
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['p0_setup_forward_attempts'], 0)
        self.assertEqual(report['s1_forward_attempts'], 0)
        self.assertIn('Checkpoint file hash mismatch', report['error'])
        self.assertTrue((self.root / 'incomplete' / 'report.json').is_file())


if __name__ == '__main__':
    unittest.main()
