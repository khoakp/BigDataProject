import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import duckdb

from src.artifacts import file_sha256
from src.graph.equivalence import checksum
from src.graph.model_replay import ModelGraphReplay, load_projection_snapshot, _hash_batches


def fixture(root):
    """Deliberately interleaved model IDs differ from canonical registration IDs."""
    raw = [(1,1,10,5.),(2,2,20,4.),(3,3,30,4.),(4,9,99,2.),
           (10,1,10,5.),(10,1,20,4.),(10,1,20,5.),(10,1,30,5.),
           (10,1,99,5.),(10,3,10,5.),(10,9,10,5.),(10,9,99,2.),
           (11,1,20,2.),(12,2,10,4.),(15,2,20,2.)]
    ordered = sorted((ts,u,m,r,i) for i,(ts,u,m,r) in enumerate(raw))
    events = [(eid,ts,u,m,r,src) for eid,(ts,u,m,r,src) in enumerate(ordered)]
    nodes = [(0,'user',2),(1,'movie',10),(2,'user',1),(3,'movie',20)]
    cohort = root/'cohort'
    cohort.mkdir()
    source = root/'canonical.parquet'
    with duckdb.connect() as con:
        con.execute('CREATE TABLE e(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating DOUBLE,source_row_id BIGINT)')
        con.executemany('INSERT INTO e VALUES (?,?,?,?,?,?)', events)
        con.execute('COPY e TO ? (FORMAT PARQUET)', [str(source)])
        con.execute('CREATE TABLE n(node_id BIGINT,node_type VARCHAR,raw_id BIGINT)')
        con.executemany('INSERT INTO n VALUES (?,?,?)', nodes)
        con.execute('COPY n TO ? (FORMAT PARQUET)', [str(cohort/'nodes.parquet')])
        con.execute('CREATE TABLE g(user_node_id BIGINT,movie_node_id BIGINT,first_event_ts BIGINT)')
        con.executemany('INSERT INTO g VALUES (?,?,?)', [(2,1,1),(0,3,2)])
        con.execute('COPY g TO ? (FORMAT PARQUET)', [str(cohort/'edges.parquet')])
    manifest = dict(cutoff_ts=10,positive_threshold=4.0,positive_edges=2,
                    graph_semantics='complete_induced_pre_cutoff_positive',
                    nodes_sha256=file_sha256(cohort/'nodes.parquet'),
                    edges_sha256=file_sha256(cohort/'edges.parquet'))
    (cohort/'manifest.json').write_text(json.dumps(manifest))
    return cohort,source,events,nodes


def reference(events,nodes,boundary):
    """Independent Python full scan, no replay SQL or normalization helper."""
    mapping = {(kind,raw):node for node,kind,raw in nodes}
    edges = {}
    for eid,ts,u,m,r,_ in events:
        if ts < boundary and r>=4 and ('user',u) in mapping and ('movie',m) in mapping:
            edges.setdefault((mapping['user',u],mapping['movie',m]),eid)
    degree = Counter(n for pair in edges for n in pair)
    coo = sorted((a,b,1/math.sqrt(degree[a]*degree[b]))
                 for u,m in edges for a,b in ((u,m),(m,u)))
    return sorted((*pair,eid) for pair,eid in edges.items()),sorted(degree.items()),coo


class ModelReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cohort,self.source,self.events,self.nodes = fixture(self.root)
        self.db = self.root/'projection.duckdb'

    def tearDown(self):
        self.tmp.cleanup()

    def create(self):
        return ModelGraphReplay.create(self.db,self.cohort,self.source)

    def test_full_reference_equivalence_boundaries_and_snapshot_reload(self):
        with self.create() as graph:
            mapping_hash = graph.meta['mapping_sha256']
            initial_mapping = graph.state()['registry']
            for boundary in (10,11,12,13,15,16):
                record = graph.advance(boundary)
                state = graph.state()
                edges,degrees,coo = reference(self.events,self.nodes,boundary)
                self.assertEqual(state['edges'],edges)
                self.assertEqual(state['degrees'],degrees)
                self.assertEqual(state['registry'],initial_mapping)
                self.assertEqual(record['canonical_watermark_event_id'],max(e[0] for e in self.events if e[1]<boundary))
                out = self.root/f'snapshot-{boundary}'
                graph.export_snapshot(out,expected_version=record['projection_version'])
                m,loaded,adj = load_projection_snapshot(out,expected_mapping=mapping_hash,expected_version=record['projection_version'],expected_boundary=boundary,expected_source=file_sha256(self.source))
                self.assertEqual(loaded,state)
                self.assertEqual(list(adj.entries),coo)
                self.assertIsNone(m['canonical_graph_version'])
                self.assertLess(m['last_canonical_order_key'][0],boundary)
            # Old exports retain old state after replay has advanced.
            m,old,_ = load_projection_snapshot(self.root/'snapshot-10',expected_mapping=mapping_hash,expected_version=0,expected_boundary=10,expected_source=file_sha256(self.source))
            self.assertEqual(len(old['edges']),2)
        with ModelGraphReplay(self.db) as reopened:
            self.assertEqual(reopened.state(),state)
            self.assertEqual(reopened.advance(11)['status'],'NO_OP')
            self.assertEqual(reopened.current()['observed_before_ts'],16)

    def test_exclusion_partition_ties_negative_events_and_empty_interval(self):
        with self.create() as graph:
            self.assertEqual(graph.current()['edge_count'],2)  # timestamp=10 excluded
            r = graph.advance(11)
            c = r['counts']
            self.assertEqual([c[k] for k in ('cold_user','omitted_user','cold_movie','omitted_movie','seen','inserted')],[1]*6)
            self.assertEqual((c['rating_events'],c['positive_events'],c['unique_positive_pairs'],c['within_interval_repeat_events']),(8,7,6,1))
            self.assertEqual(graph.advance(12)['counts']['inserted'],0)
            self.assertEqual(graph.current()['edge_count'],3)
            graph.advance(13)
            r = graph.advance(15)
            self.assertEqual(r['counts']['rating_events'],0)
            self.assertEqual(r['edge_count'],4)
            self.assertEqual(graph.advance(16)['edge_count'],4) # low rating never removes edge

    def test_retry_rollback_and_invalid_boundary(self):
        with self.create() as graph:
            before = graph.state()
            original = graph._record
            def fail_after_record(*args):
                original(*args)
                raise RuntimeError('injected after edge + version writes')
            with patch.object(graph,'_record',side_effect=fail_after_record):
                with self.assertRaises(RuntimeError):
                    graph.advance(11)
            self.assertEqual(graph.state(),before)
            self.assertEqual(graph.current()['projection_version'],0)
            graph.advance(11)
            after = graph.state()
            self.assertEqual(graph.advance(11)['status'],'NO_OP')
            self.assertEqual(graph.state(),after)
            for boundary in (9,17,True,11.5):
                with self.assertRaises(ValueError):
                    graph.advance(boundary)
            with self.assertRaises(ValueError):
                graph.export_snapshot(self.root/'wrong',expected_version=0)

    def test_source_and_database_mapping_tampering_rejected(self):
        with self.create() as graph:
            graph.con.execute("UPDATE nodes SET raw_id=888 WHERE node_id=0")
            with self.assertRaisesRegex(ValueError,'mapping'):
                graph.advance(11)
        with self.assertRaises(ValueError):
            ModelGraphReplay(self.db)
        # Source content, not a user-provided filename, is the lineage contract.
        self.source.write_bytes(self.source.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'source changed'):
            ModelGraphReplay(self.db)

    def test_cohort_hash_and_semantic_mismatch_rejected(self):
        edge_path = self.cohort/'edges.parquet'
        with duckdb.connect() as con:
            con.execute('COPY (SELECT 2::BIGINT user_node_id,3::BIGINT movie_node_id,1::BIGINT first_event_ts) TO ? (FORMAT PARQUET)', [str(self.root/'bad.parquet')])
        edge_path.write_bytes((self.root/'bad.parquet').read_bytes())
        with self.assertRaisesRegex(ValueError,'hash'):
            self.create()
        p = self.cohort/'manifest.json'
        manifest = json.loads(p.read_text())
        manifest['edges_sha256'] = file_sha256(edge_path)
        p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'Base projection'):
            self.create()

    def test_snapshot_mismatch_and_corrupt_normalization_rejected(self):
        with self.create() as graph:
            out = self.root/'snapshot'
            m = graph.export_snapshot(out,expected_version=0)
            mapping = graph.meta['mapping_sha256']
            for kwargs in ({'expected_mapping':'wrong','expected_version':0},
                           {'expected_mapping':mapping,'expected_version':99}):
                with self.assertRaises(ValueError):
                    load_projection_snapshot(out,**kwargs,expected_boundary=10,expected_source=file_sha256(self.source))
            # Even updated outer file hashes cannot hide wrong normalization.
            with duckdb.connect() as con:
                entries = con.execute('SELECT source,target,weight*0.5 FROM read_parquet(?) ORDER BY ALL',[str(out/'adjacency.parquet')]).fetchall()
                con.execute('CREATE TABLE a(source BIGINT,target BIGINT,weight DOUBLE)')
                con.executemany('INSERT INTO a VALUES (?,?,?)',entries)
                con.execute('COPY a TO ? (FORMAT PARQUET)',[str(self.root/'corrupt.parquet')])
            (out/'adjacency.parquet').write_bytes((self.root/'corrupt.parquet').read_bytes())
            m['files_sha256']['adjacency.parquet'] = file_sha256(out/'adjacency.parquet')
            m['coo_sha256'] = checksum(entries)
            (out/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'normalization'):
                load_projection_snapshot(out,expected_mapping=mapping,expected_version=0,expected_boundary=10,expected_source=file_sha256(self.source))

    def test_snapshot_temporal_provenance_and_file_integrity(self):
        with self.create() as graph:
            out = self.root/'snapshot'
            manifest = graph.export_snapshot(out,expected_version=0)
            args = dict(expected_mapping=graph.meta['mapping_sha256'],expected_version=0,
                        expected_boundary=10,expected_source=file_sha256(self.source))
            for field,value in (('observed_before_ts',11),('source_sha256','wrong'),
                                ('last_canonical_order_key',[10,1,20,4]),
                                ('snapshot_id','wrong'),('canonical_graph_version',0),
                                ('canonical_watermark_event_id',-1)):
                altered = dict(manifest,**{field:value})
                (out/'manifest.json').write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    load_projection_snapshot(out,**args)
            (out/'manifest.json').write_text(json.dumps(manifest))
            p = out/'edges.parquet'
            p.write_bytes(p.read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError,'hash'):
                load_projection_snapshot(out,**args)
            with self.assertRaises(FileExistsError):
                graph.export_snapshot(out,expected_version=0)

    def test_noncanonical_source_and_existing_database_rejected(self):
        with self.create():
            pass
        with self.assertRaises(FileExistsError):
            self.create()
        with duckdb.connect() as con:
            con.execute('CREATE TABLE bad AS SELECT event_id+1 AS event_id,* EXCLUDE(event_id) FROM read_parquet(?)',
                        [str(self.source)])
            con.execute('COPY bad TO ? (FORMAT PARQUET)',[str(self.root/'bad_source.parquet')])
        with self.assertRaisesRegex(ValueError,'canonical'):
            ModelGraphReplay.create(self.root/'bad.duckdb',self.cohort,self.root/'bad_source.parquet')

    def test_streaming_hash_matches_existing_checksum(self):
        rows=[(1,'user',3),(2,'phim',5),(3,4,0.123456789)]
        self.assertEqual(_hash_batches([rows[:1],[],rows[1:]]),checksum(rows))
        self.assertEqual(_hash_batches([]),checksum([]))

    def test_rounding_tolerance_preserves_exported_file_integrity(self):
        with self.create() as graph:
            out=self.root/'rounding'
            m=graph.export_snapshot(out,expected_version=0)
            args=dict(expected_mapping=graph.meta['mapping_sha256'],expected_version=0,
                      expected_boundary=10,expected_source=file_sha256(self.source),materialize=False)
            with duckdb.connect() as con:
                entries=con.execute('SELECT * FROM read_parquet(?) ORDER BY ALL',[str(out/'adjacency.parquet')]).fetchall()
                rounded=[(a,b,math.nextafter(w,math.inf)) for a,b,w in entries]
                con.execute('CREATE TABLE a(source BIGINT,target BIGINT,weight DOUBLE)')
                con.executemany('INSERT INTO a VALUES (?,?,?)',rounded)
                con.execute('COPY a TO ? (FORMAT PARQUET)',[str(self.root/'rounded.parquet')])
            (out/'adjacency.parquet').write_bytes((self.root/'rounded.parquet').read_bytes())
            with self.assertRaisesRegex(ValueError,'hash'):
                load_projection_snapshot(out,**args)
            m['files_sha256']['adjacency.parquet']=file_sha256(out/'adjacency.parquet')
            m['coo_sha256']=checksum(rounded)
            self.assertNotEqual(m['coo_sha256'],m['python_coo_sha256'])
            (out/'manifest.json').write_text(json.dumps(m))
            loaded,state,adj=load_projection_snapshot(out,**args)
            self.assertIsNone(state)
            self.assertIsNone(adj)
            self.assertEqual(loaded['coo_sha256'],checksum(rounded))


if __name__ == '__main__':
    unittest.main()
