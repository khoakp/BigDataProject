import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graph.incremental import IncrementalGraph
from src.graph.snapshot import materialize_normalized, load_snapshot, export_snapshot, load_export, freshness
from test_incremental import batch


class SnapshotTest(unittest.TestCase):
    def test_deferred_versions_and_export_survive_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a = batch(base,'a',[(0,10,1,1,4,0)])
            b = batch(base,'b',[(1,11,1,2,4,1)])
            db = base / 'state.duckdb'
            with IncrementalGraph(db) as graph:
                graph.apply_batch(0,a)
                materialize_normalized(graph,expected_version=1,provenance={'dataset_id':'toy'})
                old = load_snapshot(graph,1)
                self.assertEqual(old[2].entries,((0,1,1.0),(1,0,1.0)))
                self.assertEqual(materialize_normalized(graph,expected_version=1,provenance={'dataset_id':'toy'}),old[0])
                with self.assertRaisesRegex(ValueError,'different content'):
                    materialize_normalized(graph,expected_version=1,provenance={'dataset_id':'different'})
                graph.apply_batch(1,b)
                self.assertEqual(load_snapshot(graph,1),old)
                self.assertEqual(freshness(graph,1),{'canonical_version':2,'watermark_event_id':1,'max_event_ts':11,
                    'model_graph_version':1,'model_watermark_event_id':0,'staleness_batches':1,'staleness_events':1})
                with self.assertRaisesRegex(ValueError,'current canonical'):
                    materialize_normalized(graph,expected_version=1,provenance={})
                with self.assertRaisesRegex(ValueError,'Unknown'):
                    load_snapshot(graph,2)
                materialize_normalized(graph,expected_version=2,provenance={'dataset_id':'toy'})
                current = load_snapshot(graph,2)
                self.assertEqual(current[2].shape,(3,3))
                self.assertAlmostEqual(current[2].entries[0][2],1/math.sqrt(2))
                self.assertEqual(load_snapshot(graph,1),old)
                export_snapshot(graph,1,base / 'export')
                loaded = load_export(base / 'export')
                self.assertEqual(loaded[1:],old[1:])
                self.assertEqual(loaded[0]['canonical_version'],1)
                self.assertEqual(loaded[0]['model_checkpoint_version'],None)
                with self.assertRaises(FileExistsError):
                    export_snapshot(graph,1,base / 'export')
            with IncrementalGraph(db) as graph:
                self.assertEqual(load_snapshot(graph,1),old)
                self.assertEqual(load_snapshot(graph,2),current)
            with (base / 'export' / 'adjacency.parquet').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_export(base / 'export')

    def test_empty_snapshot_and_failed_materialization_rollback(self):
        with IncrementalGraph() as graph:
            materialize_normalized(graph,expected_version=0,provenance={})
            self.assertEqual(load_snapshot(graph,0)[2].shape,(0,0))
            with tempfile.TemporaryDirectory() as empty_tmp:
                export_snapshot(graph,0,Path(empty_tmp) / 'empty')
                self.assertEqual(load_export(Path(empty_tmp) / 'empty')[2].shape,(0,0))
            original = graph.con
            class FailWrite:
                def execute(self,sql,*args):
                    if sql.startswith('INSERT INTO model_graph_coo'):
                        raise RuntimeError('injected snapshot failure')
                    return original.execute(sql,*args)
            with tempfile.TemporaryDirectory() as tmp:
                a = batch(Path(tmp),'a',[(0,10,1,1,4,0)])
                graph.apply_batch(0,a)
                before = graph.snapshot()
                graph.con = FailWrite()
                try:
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        materialize_normalized(graph,expected_version=1,provenance={})
                finally:
                    graph.con = original
                self.assertEqual(graph.snapshot(),before)
                with self.assertRaisesRegex(ValueError,'Unknown'):
                    load_snapshot(graph,1)
                materialize_normalized(graph,expected_version=1,provenance={})
                graph.con.execute('UPDATE model_graph_coo SET weight=0.5 WHERE model_graph_version=1')
                with self.assertRaisesRegex(ValueError,'payload'):
                    load_snapshot(graph,1)
